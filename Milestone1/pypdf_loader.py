from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("Think_Straight.pdf")
docs =  loader.load()
print(docs[0].metadata)
print(type(docs))