

# END TO END MACHINE LEARNING PROJECT WITH AWS EC2 DEPLOYMENT

## Setup & Installation
- Python 3.7+  
- Create and activate virtual environment:  
  ```bash
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  ```
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

## Directory Structure
```
ML_Project/
├── data/          
├── notebooks/     
├── scripts/       
├── results/       
├── application.py   # Flask app for model serving
├── model.pkl        # Serialized model
├── requirements.txt
├── Dockerfile       # (Optional) Container build
└── README.md
```

## ML Workflow
- Data prep and cleaning  
- EDA and feature engineering  
- Model training and evaluation  
- Serialize model (model.pkl)  
- Flask API (`application.py`) for serving predictions

## AWS EC2 Deployment Steps
1. Launch EC2 instance (Amazon Linux or Ubuntu) with SSH and HTTP ports open.  
2. SSH into instance:  
   ```bash
   ssh -i key.pem ec2-user@
   ```
3. Install Python3, pip, and create a virtual environment.  
4. Upload code and model files to EC2 (using `scp` or SFTP).  
5. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
6. Run Flask app:  
   ```bash
   python3 application.py
   ```
7. Access at `http://:5000`

## Tips
- Ensure Flask runs on `host='0.0.0.0'` to be publicly accessible.  
- Use Elastic IP to keep a fixed EC2 IP.  
- Secure your instance with proper security group rules.

## Sample Flask app (`application.py`)
```python
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Contributing
Fork, branch, commit, push, and create PR.

## License
MIT License. See LICENSE.

This concise README is ready to help anyone deploy your ML app on AWS EC2 with `application.py` as the Flask entrypoint.
