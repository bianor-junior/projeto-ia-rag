import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from triagem import Triagem
from triagem_prompt import TRIAGEM_PROMPT
from read_pdfs import retriever, prompt_rag
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict, List,  TypedDict, Optional
import re, pathlib


load_dotenv()
google_api_key = os.getenv("GOOGLE_IA_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=google_api_key)
# response = llm.predict("Quem foi Albert Einstein?")
# print(response)
# response = llm.invoke("Quem foi Albert Einstein?")
# print(response.content)
triagem_chain = llm.with_structured_output(Triagem)

def triagem(mensagem: str) -> dict:
    saida = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()
document_chain = create_stuff_documents_chain(llm, prompt_rag)

# Formatadores
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1:
            break
    if pos == -1:
        pos = 0
    ini, fim = max(0, pos - janela // 2), min(len(txt), pos + janela // 2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source", "")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({
            "documento": src,
            "pagina": page,
            "trecho": extrair_trecho(d.page_content, query)
        })
    return cites[:3]

def perguntar_rag(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {
            "answer": "Não sei.",
            "citacoes": [],
            "contexto_encontrado": False
        }

    answer = document_chain.invoke({
        "input": pergunta,
        "context": docs_relacionados
    })

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {
            "answer": "Não sei.",
            "citacoes": [],
            "contexto_encontrado": False
        }

    return {
        "answer": txt,
        "citacoes": formatar_citacoes(docs_relacionados, pergunta),
        "contexto_encontrado": True
    }

# print('---------MENSAGEM--------')
# mensagem = "Gostaria de uma exceção para trabalhar 5 dias remoto."
# print(mensagem)
# print('---------RESULTADO--------')
# resultado = perguntar_rag(mensagem)
# print(resultado)
class AgenteState(TypedDict, total=False):
		pergunta: str
		triagem: dict
		resposta: Optional[str]
		citacoes: List[dict]
		rag_sucesso: bool
		acao_final: str

def node_triagem(state: AgenteState) -> AgenteState:
		print("-----NODE_TRIAGEM-----")
		return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgenteState) -> AgenteState:
		print("-----NODE_AUTO_RESOLVER-----")
		rag_result = perguntar_rag(state["pergunta"])
		update: AgenteState = {
			"resposta": rag_result["answer"],
			"citacoes": rag_result["citacoes", []],
			"rag_sucesso": rag_result["contexto_encontrado"],
		}
		if rag_result["contexto_encontrado"]:
			update["acao_final"] = "AUTO_RESOLVER"
		
		return update

def node_pedir_info(state: AgenteState) -> AgenteState:
		print("-----NODE_PEDIR_INFO-----")
		falt = state["triagem"].get("campos_faltantes", [])
		detalhes = ", ".join(falt) if falt else "Tema e contexto específico"
		return {
			"resposta": f"Para avançar, preciso de mais detalhes: {detalhes}",
			"citacoes": [],
			"acao_final": "PEDIR_INFO",
		}

def node_abrir_chamado(state: AgenteState) -> AgenteState:
		print("-----NODE_ABRIR_CHAMADO-----")
		triagem = state["triagem"]
		return {
			"resposta": f"Entendido. Abrirei um chamado com urgência {triagem.get('urgencia')}. Descrição: {state['pergunta'][:140]}",
			"citacoes": [],
			"acao_final": "ABRIR_CHAMADO",
		}

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgenteState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"

def decidir_pos_auto_resolver(state: AgenteState) -> str:
    print("Decidindo após o auto_resolver...")

    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"

    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"

# CRIANDO O GRAFO
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgenteState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()

# EXIBINDO O GRAFO
# Em vez de usar display() (funciona apenas em Jupyter), salvamos o PNG em disco e tentamos abrir com o visualizador padrão.
try:
    graph_bytes = grafo.get_graph().draw_mermaid_png()
except Exception as e:
    print("Erro ao gerar bytes do grafo:", e)
else:
    out_path = pathlib.Path("graph.png")
    try:
        with open(out_path, "wb") as f:
            f.write(graph_bytes)
        print(f"Grafo salvo em: {out_path.resolve()}")
    except Exception as e:
        print("Erro ao salvar o arquivo de imagem:", e)

    # Tentar abrir automaticamente (pode falhar em servidores sem interface gráfica)
    try:
        import subprocess, sys
        if sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(out_path)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(out_path)], check=False)
        else:
            print("Abra o arquivo manualmente com um visualizador de imagens.")
    except Exception as e:
        print("Não foi possível abrir automaticamente:", e)

perguntas = [
	"Gostaria de uma exceção para trabalhar 5 dias remoto.",
	"Quantas capivaras cabem em um fusca?",		
]

for msg_test in perguntas:
    resposta_final = grafo.invoke({"pergunta": msg_test})

    triag = resposta_final.get("triagem", {})
    print(f"PERGUNTA: {msg_test}")
    print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"RESPOSTA: {resposta_final.get('resposta')}")
    if resposta_final.get("citacoes"):
        print("CITAÇÕES:")
        for citacao in resposta_final.get("citacoes"):
            print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
            print(f"   Trecho: {citacao['trecho']}")

    print("------------------------------------")
# mensagem = "Gostaria de uma exceção para trabalhar 5 dias remoto."
# print(f"PERGUNTA: {mensagem}")
# resposta = perguntar_rag(mensagem)
# if resposta["contexto_encontrado"]:
#     print("RESPOSTA:", resposta["answer"])
#     print("CITAÇÕES:")
#     for c in resposta["citacoes"]:
#         print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
#         print(f"   Trecho: {c['trecho']}")
# print("------------------------------------")