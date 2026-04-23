import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), '__init__.py'),
               run_name='__main__')
