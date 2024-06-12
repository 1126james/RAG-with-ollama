Installation
1. Open cmd
2. run "ollama run llama3:latest"
3. wait for auto installation
4. run "ollama pull nomic-embed-text"
5. cd to folder directory
6. run "pip install -r requirements.txt"

By this point, all python functions should work properly.
Run the following command in the directory terminal to populate chroma db.

python new_populate_database.py

This will initialize a local chroma db, split the pdf files into chunks of text with unique ID (Format like source_file_name:page:chunk_id), and populate the chroma db with the chunks.
To make sure the repository can interact with llama3 properly, test it with query_data.py

python query_data "input_your_question_here"

If everything goes fine, wait for it to load up for the answer.
If not, good luck debugging.
Run the following to check if everything is working as expected.

pytest

if 2 cases passed, then ez.
edit the test cases inside test_rag.py 
