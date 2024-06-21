# Streamlit UI setup and handling the query to get the response from the model

import streamlit as st
from LLM import get_few_shot_db_chain

# Function to handle the query and get the response
def handle_query(question):
    chain = get_few_shot_db_chain()
    answer, sql_query = chain.run(question)

    return answer, sql_query

# Streamlit UI setup
st.title("QueryCraft AI: Database Q&A 🤖 ")


# User input box
question = st.text_input("Enter your query:")

# When the 'Send' button is clicked
if st.button("Send"):
    if question:
        response, sql_query = handle_query(question)
        
        # Displaying the SQL query and the answer
        st.write("### SQL Query")
        st.code(sql_query)
        
        st.write("### Answer")
        st.write(response)
    else:
        st.write("Please Enter a query.")
