"""
Database utilities for interacting with the CourseHub database.
"""
import psycopg2
from decouple import config

def init_db():
    """
    Initializes a connection to the CourseHub PostgreSQL database.

    Returns:
        conn (psycopg2.extensions.connection): A connection object if the connection is successful, None otherwise.
    """
    try:
        conn = psycopg2.connect(
            host=config("DB_HOST"),
            database=config("DB_NAME"),
            user=config("DB_USER"),
            password=config("DB_PASSWORD"),
            port=config("DB_PORT"),
        )
        print("Connected to the database successfully!")
        return conn
    except Exception as e:
        print("Error connecting to the database:", e)
        return None


def fetch_courses():
    """
    Fetches all courses from the CourseHub database.

    Returns:
        list[dict]: A list of dictionaries containing course data if the query is successful, None otherwise.
    """
    try:
        conn = init_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                course.id,
                course.name,
                course.slug,
                course.about,
                course.tags,
                core_institute.name as institute
            FROM
                core_course as course
            INNER JOIN
                core_institute
            ON
                core_institute.id = course.institute_id
        """)
        rows = cursor.fetchall()
        print("Number of data fetched: ", len(rows))
        columns = [c[0] for c in cursor.description]
        data_dict = [dict(zip(columns, row)) for row in rows]
        return data_dict
    except Exception as e:
        print("Error fetching data from the database:", e)
        return None


if __name__ == "__main__":
    courses = fetch_courses()
    print(courses)
