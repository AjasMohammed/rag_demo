"""
Main entry point for the Qdrant-based knowledge base.

This script initializes a Qdrant database with course data from the database,
and then enters a loop where it takes user queries and generates responses
based on the relevant documents in the database.

The responses are generated using the Gemini language model.
"""

# import configuration variables
from decouple import config

# import the database functions
from db import fetch_courses

# import the language model
from llm import LLM

# import the Qdrant database functions
from q_drant import QdrantDB


def main():
    """
    Main entry point for the script.
    """
    # set up the Qdrant database
    google_api = config("GOOGLE_API_KEY")
    collection_name = "knowledge_base"
    qdb = QdrantDB(collection_name)
    llm = LLM(model="gemini-2.0-flash", api_key=google_api)

    # create the collection if it doesn't exist
    collection_created = qdb.create_collections()
    if collection_created:
        # fetch the course data from the database
        course_dict = fetch_courses()
        # embed the course data into vectors
        texts = [f"{course['name']} {course['about']}" for course in course_dict]
        embeddings = list(qdb.embed_texts(texts))
        # create the points for the Qdrant database
        points = qdb.create_points(embeddings, course_dict)
        # ingest the points into the Qdrant database
        operation_info = qdb.ingest_data(points)
        print("Operation info: ", operation_info)
        print('Initialized Qdrant Database!')

    # enter a loop where we take user queries and generate responses
    while True:
        query = input("Query: ")
        if query == "exit":
            break
        # embed the query into a vector
        query_embeddings = next(qdb.embed_texts(query))
        # search the Qdrant database for relevant documents
        results = qdb.query_collections(query_embeddings)
        # parse the results into a string
        parsed_results = "\n".join([
            f"""
                name: {result.payload['name']}
                description: {result.payload['about']}
                institute: {result.payload['institute']}
            """
            for result in results
        ])
        # use the language model to generate a response
        llm_query = llm.invoke(query, parsed_results)
        # print the response
        print(f"AI: {llm_query.content}")


if __name__ == "__main__":
    main()

