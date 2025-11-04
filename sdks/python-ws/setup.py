"""Setup script for Realm WebSocket Client SDK"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="realm-ws-client",
    version="0.1.0",
    author="Realm.ai",
    description="WebSocket client SDK for Realm.ai multi-tenant LLM inference server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/realm-ai/realm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "websockets>=11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
)

