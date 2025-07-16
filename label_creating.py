# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model("Model/ASL_2.h5")
#
# # Get the class labels (assuming it's a classification model)
# class_labels = model.class_names if hasattr(model, "class_names") else list(range(model.output_shape[-1]))
#
# # Save labels to a text file
# with open("labels.txt", "w") as f:
#     for label in class_labels:
#         f.write(f"{label}\n")
#
# print("labels.txt has been created.")