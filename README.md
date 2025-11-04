# Verificador de Identidad por Imagen

**Proyecto Práctico 2 – Ingeniería de Software / Inteligencia Artificial**

Este proyecto implementa un sistema de verificación facial personal, capaz de responder si una imagen corresponde o no al propio rostro del usuario.  
El sistema utiliza embeddings faciales preentrenados, un clasificador binario (Logistic Regression o SVM) y una API REST con Flask desplegada en AWS EC2.

---

## Objetivo

Entrenar un modelo que responda a la pregunta:

> “¿Soy yo?”

A partir de imágenes faciales, el modelo entrega una respuesta booleana (`True`/`False`) y una confianza numérica (score).

---

## Tecnologías utilizadas

- **Python 3.11+**
- **Flask**
- **PyTorch**
- **facenet-pytorch** 
- **scikit-learn**
- **Pillow**
- **NumPy**
- **joblib**
- **dotenv**
- **gunicorn**
- **AWS EC2 (Ubuntu 22.04)**

---
