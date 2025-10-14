# save_savedmodel_legacy.py
import argparse, tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

# Load Keras 3 model
model = tf.keras.models.load_model(args.ckpt, compile=False)
# Save as TensorFlow SavedModel (graph form)
tf.saved_model.save(model, args.outdir)
print("SavedModel written to:", args.outdir)
