import onnx
# 加载模型
model = onnx.load('Feathernet_SE_IR.onnx')

node  = model.graph.node
out = model.graph.output

del out[1]
onnx.save(model, 'Feathernet_SE_IR_new.onnx')
print("OK")
