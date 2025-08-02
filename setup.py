from setuptools import find_packages, setup

setup(
    name='info-retrieval-system',
    version='0.1.0',
    author='SHREEVISHNU',
    author_email='shreevishnusreekanth@gmail.com',
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'langchain-google-genai',
        'google-generativeai',
        'langchain',
        'PyPDF2',
        'faiss-cpu',
        'streamlit'
    ]
)
