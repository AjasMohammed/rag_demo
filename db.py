"""
Database utilities for interacting with the CourseHub database.
"""
import psycopg2
from base import BaseRAGDBClient


class RAGPGClient(BaseRAGDBClient):
    def __init__(self, host="localhost", database="postgres", user="postgres", password="postgres", port=5432):
        self.conn = self.connect(host, database, user, password, port)
        self.cursor = self.conn.cursor() if self.conn else None

    def connect(self, host, database, user, password, port):
        try:
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
            )
            return conn
        except Exception as e:
            print("Error connecting to the database:", e)
            return None

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def fetch_named_data(self, query: str) -> list[dict]:
        """
        Fetches relevant data from the database.

        Returns:
            list[dict]: A list of dictionaries containing course data if the query is successful, None otherwise.
        """
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            print("Number of data fetched: ", len(rows))
            columns = [c[0] for c in self.cursor.description]
            data_dict = [dict(zip(columns, row)) for row in rows]
            return data_dict
        except Exception as e:
            print("Error fetching data from the database:", e)
            return None

if __name__ == "__main__":
    pass
