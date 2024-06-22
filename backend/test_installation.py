try:
    from flask import Flask, request, jsonify
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch
    import io

    print("All libraries are installed correctly.")
except ImportError as e:
    print(f"Error: {e}")