#!/usr/bin/python
from http.server import BaseHTTPRequestHandler,HTTPServer
import os
import json
import urllib
import pandas

PORT_NUMBER = 8080

# HTML elements builders

class Html:
	def __init__(self):
		self.body = []

	def add(self, elem):
		self.body.append(elem.str())

	def _style(self):
		return """
		<style>
			table { 
				margin: 10px;
				margin-right: 20px;
				border-collapse: collapse; 
			}
			table, th, td { 
				border: 1px solid; 
			}

			th {
				text-align: center;
				background-color: black;
				color: white;
			}

			td :not(pre) {
				padding-left: 10px;
				padding-right: 5px;
				text-align: right;
			}

			pre {
				border: 1px solid black;
				background-color: rgb(240, 240, 240);
			}

			.result_cell {
				overflow: auto;
				height: 300px;
			}
		</style>
		"""

	def str(self):
		return ("<html>"
			"<head>" +
			self._style() +
			"</head>"
			"<body>" + "\n".join(self.body) + "</html><body>")

class Ul:
	def __init__(self):
		self.items = []

	def add(self, elem):
		self.items.append("<li>" + elem.str() + "</li>")

	def str(self):
		return "<ul>" + "\n".join(self.items) + "</ul>"

class A:
	def __init__(self, text, url):
		self.a = "<a href=\"" + url + "\">"+text+"</a>"

	def str(self):
		return self.a

class H:
	def __init__(self, level, text):
		self.h = "<h" + str(level) + ">" + text +"</h" + str(level) +">"

	def str(self):
		return self.h

class P:
	def __init__(self, text):
		self.p = "<p>" + text +"</p>"

	def str(self):
		return self.p

class Pre:
	def __init__(self, text):
		self.pre = "<pre>" + text +"</pre>"

	def str(self):
		return self.pre

class Div:
	def __init__(self, elem, klass=""):
		self.div = "<div"
		if klass != "":
			self.div += " class=\"" + klass +"\""
		self.div += ">" + elem.str() + "</div>"

	def str(self):
		return self.div

class Str:
	def __init__(self, str):
		self.s = str

	def str(self):
		return self.s

class Table:
	def __init__(self):
		self.table = []

	def add_header(self, row):
		self.table.append( " ".join(map(lambda x : "<th>" + x.str() + "</th>", row)) )

	def add(self, row):
		self.table.append( " ".join(map(lambda x : "<td>" + x.str() + "</td>", row)) )

	def str(self):
		return "<table>" + "\n".join(map(lambda x : "<tr>" + x + "</tr>", self.table)) + "</table>"



#  Actual logic for retrieving pages

class ResultRequestHandler(BaseHTTPRequestHandler):

	#Handler for the GET requests
	def do_GET(self):
		self.send_response(200)
		self.send_header('Content-type','text/html')
		self.end_headers()
		
		if self.path == "/":
			self.wfile.write(self.__get_index_html().encode('utf-8'))
		else:
			self.wfile.write(self.__get_result_html(self.path).encode('utf-8'))

		return

	def __get_index_html(self):
		result = Html()
		ul = Ul()

		for dirname in [d for d in os.listdir(".") if os.path.isdir(d) and d[0] != '.']:
			ul.add(A(dirname,dirname))

		result.add(ul)
		return result.str()
	
	def __diff_conf(self, conf, base_conf):
		result = {}
		for key in conf.keys():
			if key not in base_conf:
				result[key] = ("+", conf[key])
				continue

			if isinstance(base_conf[key], dict):
				diff = self.__diff_conf(conf[key], base_conf[key])
				if diff != {}:
					result[key] = diff
				continue

			if base_conf[key] != conf[key]:
				result[key] = ("!=", conf[key])
				continue
		
		for key in base_conf.keys():
			if key not in conf:
				result[key] = ("-", base_conf[key])

		return result

		

	def __get_result_html(self, path):
		dir = urllib.parse.unquote(os.path.relpath(path, "/"))
		base_conf = None

		result = Html()
		result.add(H(1, dir))
		table = Table()
		table.add_header([Str("Configuration diff"), Str("Results (Opt)"), Str("Results (Alt)")])

		innerdirs = [os.path.join(dir,d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d)) and d[0] != '.']
		for d in innerdirs:
			conf_file = os.path.join(d, "conf.json")
			if not os.path.isfile(conf_file):
				continue

			with open(conf_file) as f:
				conf = json.load(f)

			if base_conf == None:
				base_conf = conf
				result.add(H(2, "Base configuration"))
				result.add(Pre(json.dumps(conf, indent=2)))

			conf = self.__diff_conf(conf, base_conf)

			out_conf = Pre(json.dumps(conf, indent=2))
			out_results_opt = Div( Str(self.__get_result(os.path.join(d, "results-OPT.csv"))), klass="result_cell")
			out_results_alt = Div( Str(self.__get_result(os.path.join(d, "results-ALT.csv"))), klass="result_cell")
			table.add( [ out_conf, out_results_opt, out_results_alt ] )

		result.add(table)
		return result.str()

	def __get_result(self, path):
		if not os.path.isfile(path):
			return ""

		pd = pandas.read_csv(path)

		return pd.to_html()


#  Main program

try:
	#Create a web server and define the handler to manage the
	#incoming request
	server = HTTPServer(('', PORT_NUMBER), ResultRequestHandler)
	print('Started httpserver on port ' , PORT_NUMBER)
	
	#Wait forever for incoming htto requests
	server.serve_forever()

except KeyboardInterrupt:
	print('^C received, shutting down the web server')
	server.socket.close()