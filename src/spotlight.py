import argparse  # To parse command line arguments
import subprocess  # To run spotlight as a subprocess

from keywords import extract_keywords
from qa import generate_answer


def search_with_spotlight(query, folder=None):
    # Prepare the command
    command = ["mdfind"]
    if folder is not None:
        command += ["-onlyin", folder]
    command += [query]

    # Execute mdfind command and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode != 0:
        print("Error in executing search")
        return []

    # Split the output into lines to get individual file paths
    file_paths = result.stdout.strip().split("\n")
    return file_paths


def chat(query, folder=None):
    # Let's extact keywords from the query
    keywords = extract_keywords(query)

    # Construct an OR query from the keywords
    search_query = " OR ".join(keywords)

    # Perform spotlight search
    search_results = search_with_spotlight(search_query, folder=folder)

    # Generate answer for the original query and top 3 search results
    print(generate_answer(query, search_results[:3], doc_type="Candidate"))

    # Print search results
    print(search_results)


if __name__ == "__main__":  # Entry point of the program
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Search files using Spotlight.")
    parser.add_argument("--folder", type=str, help="Folder to search in", default=None)
    parser.add_argument("query", type=str, help="Search query")

    # Parse commandline arguments
    args = parser.parse_args()

    chat(args.query, args.folder)
