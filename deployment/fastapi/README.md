# Deploying Your Model Locally ✨

Ready to get the model up and running locally? Follow these easy steps:

### **Create or Activate Your Pipenv Environment:**

If you haven’t set up a Pipenv environment yet, you can do so by running:

If you have already activated the pipenv/virtual environment as in the Setup you can skip this step

```bash
pip install pipenv
```

If you already have Pipenv environment installed, activate it with:

```bash
pipenv install
pipenv shell
```

This ensures all the required dependencies are installed for your project. Check the `requirements.txt` for a list of dependencies used.

### **Package the App into Docker:**

The prediction script has been embedded into a FastAPI application and packed into a Docker image. To build the Docker image, run:

```bash
docker build -t brain:v1 .
```

**Pls make sure your bash is in this folder containing the dockerfile**

### **Run the Docker Container:**

Once the Docker image is built, you can run it locally with:

```bash
docker run -it --rm -p 8000:8000 brain:v1
```

This command will start the container and expose it on port `8080`. You can access your FastAPI app at `http://localhost:8080`.

### **Use the FastAPI App for Predictions:**

To use the FastAPI app for predictions, you need to run `test.py`.

```python
python test.py
```

You can change the sample in the test.py to see the output for each value.

Feel free to check the `predict.py` file for more details on how the script and Flask app are set up. If you have any questions or need further assistance, just let me know. Happy deploying! 🎉
