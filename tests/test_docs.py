import pytest
import nbval

# You can add more notebook files to this list as needed
notebook_files = [
    'docs/doc.ipynb'
]

@pytest.mark.parametrize("notebook", notebook_files)
def test_notebook(notebook):
    pytest.main(["--nbval", notebook])

if __name__ == "__main__":
    pytest.main([__file__])