interpreter = tf.lite.Interpreter(model_path="FashionNet_model_standard.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], train_images[0:1].astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

interpreter = tf.lite.Interpreter(model_path="FashionNet_model_dynamic_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], train_images[0:1].astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

interpreter = tf.lite.Interpreter(model_path="FashionNet_model_integer_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], train_images[0:1].astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)