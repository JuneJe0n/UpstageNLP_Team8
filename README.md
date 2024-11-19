# Upstage-NLP-Project_Team8

## Install Dependencies
1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing ,`onnxruntime` through `pip install onnxruntime`. 
```python
conda install onnxruntime -c conda-forge
```

2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

## Create database

Create the Chroma DB.

```python
python populate_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "영어 및 정보 등에 관하여 일정한 기준의 능력이나 자격을 취득한 경우 인정 받는 학점은 몇점인가?"
```
or use a test.csv file to query the Chroma DB.
```python
python query_multiplechoice.py
```
