from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama


## Load local LLM
custom_url = 'http://localhost:11434'

llm = ChatOllama(
    base_url = custom_url,
    model = 'mistral'
)
print("Local LLM loaded")


## Load vector DB
persist_directory = "./vector-db"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print('Vector DB loaded')


def get_response_from_query(db, query):

    # embed query, find k nearest docs, combine docs
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding)
    docs_page_content = " ".join([d.page_content for d in docs])


    # generate LLM answer based on similar docs
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
      This bot engages in discussions on a wide range of topics, including cultural, philosophical, and political matters. It analyzes provided articles to inform its responses. Please adhere to the truth. If no resources are available, share your personal opinion.

      Question to be answered: {question}

      Referenced articles for analysis: {docs}

      Instructions for the bot:
      1. Extract and use only factual information from the specified documents.
      2. Highlight key phrases and evidence from the articles to support your answers.
      3. If the articles do not sufficiently cover the topic to provide an informed response, please state, "I don't have enough information to answer this question."

      Remember, the goal is to provide well-informed, accurate, and thoughtful responses based on the available resources. If personal opinion is necessary due to a lack of information, it should be clearly identified as such.
      """
      ,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content,return_source_documents=True)
    r_text = str(response)

    # use LLM to evaluate answer
    prompt_eval = PromptTemplate(
        input_variables=["answer", "docs"],
        template="""
          Your task is to assess whether the provided response accurately and faithfully reflects the context of a given question or statement.

          Evaluate the following response: {answer}
          Reference article for evaluation: {docs}

          Instructions for the evaluation:
          1. Start your evaluation with a clear "Yes" or "No" to indicate if the response is faithful to the context provided by the reference article.
          2. Provide a detailed reason for your judgment. Mention specific aspects of the response and the article that support your evaluation. Highlight any direct correlations, discrepancies, or notable omissions in the response compared to the factual content of the article.
          3. If the response incorporates elements not found in the article but remains relevant and truthful to the broader topic, please acknowledge this as a factor in your assessment.

          Your evaluation should focus on the accuracy, relevance, and completeness of the response in relation to the information presented in the referenced article. This ensures a thorough and reasoned assessment of the response's faithfulness to the context.
          """
          ,
    )

    chain_part_2 = LLMChain(llm=llm, prompt=prompt_eval)
    evals = chain_part_2.run(answer=r_text, docs=docs_page_content)

    return response,docs,evals


import gradio as gr

def greet(query):
    answer,sources,evals = get_response_from_query(db,query)
    return answer,sources,evals


demo = gr.Interface(fn=greet,
                    title="Local-RAG",
                    inputs=["text"],
                    outputs=[gr.components.Textbox(lines=3, label="Response"),
                             gr.components.Textbox(lines=3, label="Source"),
                             gr.components.Textbox(lines=3, label="Evaluation")],
                   )

demo.launch(share=True, debug=True)