import os
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from decimal import Decimal
from langchain.sql_database import SQLDatabase

from few_shot_prompts import few_shots
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

def get_few_shot_db_chain():
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots if example]  # Ensure non-empty examples

    if not to_vectorize:
        raise ValueError("The 'few_shots' list is empty or improperly formatted.")

    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
    )

    class ModifiedSQLDatabaseChain(SQLDatabaseChain):
        def run(self, input: str):
            prompt_input = {
                'input': input,
                'table_info': '',  # Add table info if necessary
                'top_k': str(5)  # Set a default value or pass it dynamically, converted to string
            }
            generated_sql_query = self.llm_chain(prompt_input)['text']

            # Extract only the SQLQuery part
            sql_query_start = generated_sql_query.find("SQLQuery: ") + len("SQLQuery: ")
            sql_query_end = generated_sql_query.find("SQLResult: ")
            sql_query = generated_sql_query[sql_query_start:sql_query_end].strip()

            # Run the SQL query
            sql_result = self.database.run(sql_query)
            

            # Process the result
            final_answer = self.llm_chain.output_parser.parse(sql_result)
            return final_answer, sql_query

    chain = ModifiedSQLDatabaseChain.from_llm(llm, db, verbose=False, prompt=few_shot_prompt)

    return chain

# Example usage
chain = get_few_shot_db_chain()
input_question = "How many t-shirts do we have left for Nike in XS size and white color?"
answer, generated_sql_query = chain.run(input_question)

print("Generated SQL query:", generated_sql_query)
print("Answer:", answer)