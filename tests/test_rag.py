from app.rag import SimpleRAG


def test_retrieve_empty():
    rag = SimpleRAG()
    rag.chunks = []
    assert rag.retrieve("hello") == []