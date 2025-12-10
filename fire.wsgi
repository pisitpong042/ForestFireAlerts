import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/html/fire")
 
from app_test import app as application
application.secret_key = "fhkjdskjgf"
