#!/usr/bin/python
from http.server import BaseHTTPRequestHandler,HTTPServer
import os
import json
import urllib
import pandas
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

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
				white-space: pre-wrap;
				border: 1px solid black;
				background-color: rgb(240, 240, 240);
			}

			pre span.debug {
				background-color: blue;
				color: white;
			}

			pre span.info {
				background-color: rgb(9, 128, 13);;
				color: white;
			}

			pre span.warning {
				background-color: orange;
				color: white;
			}

			pre span.error {
				background-color: red;
				color: white;
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
			"<meta charset=\"utf-8\">" +
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
		self.open_div = "<div"
		if klass != "":
			self.open_div += " class=\"" + klass +"\""
		self.open_div += ">" 
		self.elems = elem.str()

	def add(self, elem):
		self.elems += elem.str()

	def str(self):
		return self.open_div + self.elems + "</div>"

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

class Img:
	def __init__(self, src, alt):
		self.img_src = src
		self.img_alt = alt

	def str(self):
		return "<img src=\"" + self.img_src + "\" alt=\"" + self.img_alt + "\">"


#  Actual logic for retrieving pages

class ResultRequestHandler(BaseHTTPRequestHandler):

	#Handler for the GET requests
	def do_GET(self):
		# root index
		if self.path == "/":
			self.send_response(200)
			self.send_header('Content-type', 'text/html')
			self.end_headers()

			self.wfile.write(self.__get_index_html().encode('utf-8'))
		elif self.path.startswith("/images"):
		# images
			self.send_response(200)
			self.send_header('Content-type', 'image/png')
			self.end_headers()

			self.wfile.write(self.__get_image(self.path))
		elif self.path.endswith(".log"):
		# log files
			self.send_response(200)
			self.send_header('Content-type', 'text/html')
			self.end_headers()

			self.wfile.write(self.__get_log_html(self.path).encode('utf-8'))
		else:
		# result tables
			self.send_response(200)
			self.send_header('Content-type', 'text/html')
			self.end_headers()

			self.wfile.write(self.__get_result_html(self.path).encode('utf-8'))

		return

	def __get_index_html(self):
		result = Html()
		ul = Ul()

		for dirname in [d for d in os.listdir(".") if os.path.isdir(d) and d[0] != '.']:
			if dirname != "images":
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
		table.add_header([Str("Configuration diff"), Str("Results (Opt)"), Str("Results (Alt)"), Str("Graph"), Str("Log files")])

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
			out_logs = Div(P(A("results-OPT.log", os.path.join(d,"results-OPT.log")).str()))
			out_logs.add(P(A("results-ALT.log", os.path.join(d,"results-ALT.log")).str()))
			graph = Div(Img("/images/" + os.path.join(d, "results.png"), "result graphs"))
			table.add( [ out_conf, out_results_opt, out_results_alt, graph, out_logs ] )

		result.add(table)
		return result.str()

	def __format_log(self, s):
		result = re.sub("Debug:", "<span class=\"debug\">Debug:</span>", s)
		result = re.sub("Warning:", "<span class=\"warning\">Warning:</span>", result)
		result = re.sub("Info:", "<span class=\"info\">Info:</span>", result)
		result = re.sub("Error:", "<span class=\"error\">Error:</span>", result)
		return result

	def __get_log_html(self, path):
		file_path = urllib.parse.unquote(os.path.relpath(path, "/"))
		result = Html()
		result.add(H(1, file_path))
		with open(file_path, "r") as f:
			result.add(Pre(self.__format_log(f.read())))
		return result.str()

	def __get_result(self, path):
		if not os.path.isfile(path):
			return ""

		pd = pandas.read_csv(path)

		return pd.to_html()

	def __create_img_path(self, path):
		current = "images/"
		for elem in os.path.split(path):
			current = os.path.join(current, elem)
			if not os.path.isdir(current):
				os.mkdir(current)


	def __get_image(self, path):
		path = urllib.parse.unquote(os.path.relpath(path, "/"))
		data_path = re.search('images/(.*)/results.png', path).group(1)
		title = re.search('images/([^/]+)/.*',path).group(1)

		pd_alt = pandas.read_csv(os.path.join(data_path, "results-ALT.csv"))
		pd_opt = pandas.read_csv(os.path.join(data_path,"results-OPT.csv"))
		self.__create_img_path(data_path)

		plt.clf()
		plt.grid(b=True, which='major')
		plt.grid(b=True, which='minor', linestyle="--")
		plt.title(title + " dataset ")
		plt.plot(pd_alt["TimeCumulative"], pd_alt["TrainBest"], "-o", label="PartitionedLS-Alt")
		plt.plot(pd_opt["TimeCumulative"], pd_opt["TrainBest"], "o", label="PartitionedLS-Opt")
		plt.xscale("log")
		plt.yscale("log")
		plt.legend()
		plt.xlabel("Time (log scale)")
		plt.ylabel("Objective")
		plt.savefig(path, format="png")

		
		with open(path, "rb") as file:
			return file.read()
		


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
