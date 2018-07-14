from flask import Flask, jsonify, request

data = {
    'huhy': {'age': 24, 'sex': '女'},
    'liuer': {'age': 12, 'sex': '男'}
}

app = Flask(__name__)  # 创建一个服务，赋值给APP
@app.route('/get_user', methods=['get'])  # 指定接口访问的路径，支持什么请求方式get，post
def get_user():  # 讲的是封装成一种静态的接口，无任何参数传入,这里的函数名称可以任意取
    # username = request.form.get('username')  # 获取接口请求中form-data的username参数传入的值
    # username = request.json.get('username')  # 获取带json串请求的username参数传入的值
    username = request.args.get('username')  # 使用request.args.get方式获取拼接的入参数据
    print(username)
    return jsonify(data)  # 把字典转成json串返回


"""
这个host：windows就一个网卡，可以不写，而liux有多个网卡，写成0:0:0可以接受任意网卡信息,
 通过访问127.0.0.1:8802/get_user，可返回data信息
debug:调试的时候，可以指定debug=true；如果是提供接口给他人使用的时候，debug要去掉
"""
app.run(host='0.0.0.0', port=8802, debug=True)