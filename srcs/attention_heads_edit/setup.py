from setuptools import find_packages, setup

install_requires = [
    "transformers>=4.31.1",
]

setup(
    name="attention_heads_edit_lib",
    version="0.1.3",
    description=(
        "PyTorch implementation of Attention Heads Edit, a post-hoc attention steering approach "
        "that emphasizes specific contexts for LLMs."
    ),
    long_description="Attention Heads Edit library.",
    long_description_content_type="text/plain",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7.0",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
