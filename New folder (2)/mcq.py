# install libraries if needed

# !pip install openai
# !pip install unstructured
# !pip install tiktoken
# !pip install pinecone-client
# !pip install pypdf
# !pip install langchain
# !pip install sentence-transformers

# output in markdown format
markdown_text=[]

# Entire function to generate MCQ
def get_mca_questions(k): 
    
    # Importing libraries and dependencies
    import openai
    import pinecone
    from langchain.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Pinecone
    from langchain.llms import OpenAI
    from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    
    # Import Dependencies
    # The code sets environment variables for accessing OpenAI API 
    # and Hugging Face Hub API using respective API keys
    import os
    os.environ["OPENAI_API_KEY"] = "<Enter your key>"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<Enter your key>"
    
    # Load Documents
    # Loads PDF files available in a directory with pypdf
    # Function to read documents
    def load_docs(directory):
      loader = PyPDFDirectoryLoader(directory)
      documents = loader.load()
      return documents

    # Passing the directory to the 'load_docs' function
    directory = 'nlp_docs/'
    documents = load_docs(directory)
    len(documents)
    
    # Transform Documents
    # Split document Into Smaller Chunks
    # This function will split the documents into chunks
    def split_docs(documents, chunk_size=1000, chunk_overlap=20):
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      docs = text_splitter.split_documents(documents)
      return docs

    docs = split_docs(documents)
    # print(len(docs))
    
    # Generate Text Embeddings
    # Hugging Face LLM for creating Embeddings for documents/Text
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Let's test our Embeddings model for a sample text
    query_result = embeddings.embed_query("Hello Buddy")
    # len(query_result)

    # Vector store - PINECONE
    # Pinecone allows for data to be uploaded into a vector database and true semantic 
    # search can be performed.
    # Not only is conversational data highly unstructured, but it can also be complex. 
    # Vector search and vector databases allows for similarity searches.
    # We will initialize Pinecone and create a Pinecone index by passing our documents,
    # embeddings model and mentioning the specific INDEX which has to be used
    # Vector databases are designed to handle the unique structure of vector embeddings, 
    # which are dense vectors of numbers that represent text. They are used in machine learning
    # to capture the meaning of words and map their semantic meaning.
    # These databases index vectors for easy search and retrieval by comparing values and 
    # finding those that are most similar to one another, making them ideal for natural language
    # processing and AI-driven applications.

    pinecone.init(
        api_key="d2d535cb-4552-4a9c-8128-b927a0697c39",
        environment="gcp-starter"
    )

    index_name = "mcq-assess"

    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    
    # Retrieve Answers
    # This function will help us in fetching the top relevent documents from our 
    # vector store - Pinecone
    def get_similiar_docs(query, k=2):
        similar_docs = index.similarity_search(query, k=k)
        return similar_docs
    
    # 'load_qa_chain' Loads a chain that you can use to do QA over a set of documents.
    #  And we will be using Huggingface for the reasoning purpose
    from langchain.chains.question_answering import load_qa_chain
    from langchain import HuggingFaceHub
    
    # BigScience Large Open-science Open-access Multilingual Language Model (BLOOM) 
    # is a transformer-based large language model.
    # It was created by over 1000 AI researchers to provide a free large language model 
    # for everyone who wants to try. Trained on around 366 billion tokens over March through July 2022,
    # it is considered an alternative to OpenAI's GPT-3 with its 176 billion parameters.
    llm=HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    
    # Different Types Of Chain_Type:
    # "map_reduce": It divides the texts into batches, processes each batch separately with 
    # the question, and combines the answers to provide the final answer.
    # "refine": It divides the texts into batches and refines the answer by sequentially processing 
    # each batch with the previous answer.
    # "map-rerank": It divides the texts into batches, evaluates the quality of each answer from LLM,
    # and selects the highest-scoring answers from the batches to generate the final answer. These 
    # alternatives help handle token limitations and improve the effectiveness of the 
    # question-answering process.
    chain = load_qa_chain(llm, chain_type="stuff")
    
    #This function will help us get the answer to the question that we raise
    def get_answer(query):
      relevant_docs = get_similiar_docs(query)
      print(relevant_docs)
      response = chain.run(input_documents=relevant_docs, question=query)
      return response

    our_query = k
    answer = get_answer(our_query)
    # print(answer)

    

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema

    response_schemas = [
        ResponseSchema(name="question", description="Question generated from provided input text data."),
        ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
        ResponseSchema(name="Correct Options", description="Two correct options for the asked question in ampersand separated.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # This helps us fetch the instructions the langchain creates to fetch the response in 
    # desired format
    format_instructions = output_parser.get_format_instructions()

    # create ChatGPT object
    chat_model = ChatOpenAI()
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""When a text input is given by the user, please generate multiple choice questions 
            from it along with the correct options. 
            \n{format_instructions}\n{user_prompt}""")  
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )

    final_query = prompt.format_prompt(user_prompt = answer)
#     final_query.to_messages()

    final_query_output = chat_model(final_query.to_messages())
    markdown_text.append(final_query_output.content)
    print(final_query_output.content)
    
lst=['What was one of the strategies employed by Tipu Sultan during his leadership?"',
    "What did the British realise about the countryside in Europe?",
    "what were the two main systems of indigo cultivation? ","which kings ruled Mysore?"]
for q in lst:
    get_mca_questions(q)
    
# After  getting the raw results it is stored in the variable called markdown_text as a list

# Cleaning The output in markdown format
##print(markdown_text)

# Let's extract JSON data from Markdown text that we have
json_string=[]
import re
import json
for f in markdown_text:
    try:
        json_ = re.search(r'{(.*?)}',f, re.DOTALL).group(1)
        json_string.append(json_)
    except AttributeError:
        pass
        
#print(json_string)

#Extracting the question,choices and answers in separate lists
ques=[]
choice=[]
answer=[]
for z in range(len(json_string)):
    u=[i for i in json_string[z].split(',')]
    t=[]
    t1=[]
    for j in u:
        for j1 in j.split(':'):
            if j1=='\n\t"question"' or j1=='\n    "question"':
                t.append(f'Q{1+z}')
                t.append(':')
                e=j.split()
                t.append(' '.join(e[1:])+'\n')
            if j1=='\n\t"Correct Options"':
                t1.append('Correct Options')
                t1.append(':')
                e=j.split()
                e1=[]
                for n in e[-1]:
                    e1.append(n)
                t1.append(f"'({e1[1].lower()})'")
                t1.append('&')
                t1.append(f"'({e1[-2].lower()})'")
    ques.append(t)
    answer.append(t1)
    
    u1=[i for i in json_string[z].split()]
    t21=[]
    t2=[]
    for v in u1:
        if v=='"a)' or v=='"A)':
            x=u1.index(v)
            try:
                x1=u1.index('b)')
            except ValueError:
                x1=u1.index('B)')
            t2.append(' a.')
            t2.append(' '.join((u1[x+1:x1])))
            t21.append(' '.join(t2).replace(',','')+'\n')
            t2.clear()
        elif v=='b)' or v=='B)':
            x=u1.index(v)
            try:
                x1=u1.index('c)')
            except ValueError:
                x1=u1.index('C)')
            t2.append('b.')
            t2.append(' '.join((u1[x+1:x1])))
            t21.append(' '.join(t2).replace(',','')+'\n')
            t2.clear()
        elif v=='c)' or v=='C)':
            x=u1.index(v)
            try:
                x1=u1.index('d)')
            except ValueError:
                x1=u1.index('D)')
            t2.append('c.')
            t2.append(' '.join((u1[x+1:x1])))
            t21.append(' '.join(t2).replace(',','')+'\n')         
            t2.clear()
        elif v=='d)' or v=='D)':
            x=u1.index(v)
            t2.append('d.')
            t2.append(' '.join((u1[x+1:-3])))
            t21.append(' '.join(t2).replace('",',''))
            t2.clear()
    choice.append(t21)
    
import fontstyle
final=[]

for ans in answer:
    # format text
    text = fontstyle.apply(''.join(ans), 'bold/black')
 
    # display text
    final.append(text)

for w in range(len(ques)):
    print(''.join(ques[w]))
    print(' '.join(choice[w]))
    print(final[w])
    print('\n')


# We have got amazing results!!!
