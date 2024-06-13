import os
import pickle
from dotenv import load_dotenv

# Для сбора компонентов GigaChat и RAG:
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# Модуль для управления предупреждениями:
import warnings
warnings.filterwarnings("ignore")

directory = "GigaChat/documents"


# Установка переменной окружения
def set_environ():
    load_dotenv()
    os.environ["GIGACHAT_CREDENTIALS"] = os.getenv("GIGACHAT_CREDENTIALS")
    return


def uploading_file_txt(path: str):
    loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)  # cp1251
    doc_1 = loader.load()
    return doc_1


def uploading_file_pdf(path: str):
    loader = PyPDFLoader(path)
    doc_1 = loader.load()
    # with open(path, "rb") as f:
    #     doc_1 = f.read()
    return doc_1


def splitting_doc(doc_1):
    # Определяем сплиттер:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Разбиваем документ:
    split_doc_1 = text_splitter.split_documents(doc_1)
    return split_doc_1


def create_vector_storage(split_doc_1):
    # Если у вас нет видеокарты, укажите 'device': 'cpu'
    hf_embeddings_model = HuggingFaceEmbeddings(
        model_name="cointegrated/LaBSE-en-ru",
        model_kwargs={"device": "cpu"}
    )

    # Создаем FAISS индекс (базу векторов) с полученными эмбеддингами
    db_1 = FAISS.from_documents(split_doc_1, hf_embeddings_model)
    return db_1


def save_db_to_the_folder(db_1, path: str):
    file_name = path.split("/")[-1].replace(".txt", "").replace(".pdf", "")
    # Сохраняем сериализованный объект в файл
    with open(f'{directory}/{file_name}.pkl', 'wb') as f:
        pickle.dump(db_1, f)


# Соберём компоненты RAG
def create_qa_chain():
    # Инициализируем языковую модель GigaChat
    # verify_ssl_certs=False – без использования сертификатов Минцифры
    llm = GigaChat(verify_ssl_certs=False, temperature=0.01)

    count = 0
    for filename in os.listdir(directory):
        if count == 0:
            count += 1
            with open(f"{directory}/{filename}", 'rb') as file:
                db = pickle.load(file)
        else:
            with open(f"{directory}/{filename}", 'rb') as file:
                piece_db = pickle.load(file)
                db.merge_from(piece_db)

    qa_chain_1 = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    return qa_chain_1


# Получаем ответ на вопрос
def get_answer(question_1: str):
    if not os.path.exists(directory):
        return {"result": "Нет загруженных документов для составления ответа"}
    qa_chain = create_qa_chain()
    return qa_chain(question_1)


if __name__ == '__main__':
    directory = "documents"

    #  pass

    # file = uploading_file_pdf("documents_pdf/FZ187.pdf")  # documents_pdf/FSB366.pdf
    # split_file = splitting_doc(file)
    # db = create_vector_storage(split_file)
    # save_db_to_the_folder(db, "documents_pdf/FZ187.pdf")
    #
    # set_environ()
    # print(get_answer("как дела")["result"])

