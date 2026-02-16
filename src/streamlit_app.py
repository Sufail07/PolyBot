import os                                                                                                                                                                     
import logging
from datetime import datetime, timezone                                                                                                                                       
                                                                                                                                                                            
import requests                                                                                                                                                               
import streamlit as st                                                                                                                                                        
                                                                                                                                                                            
logging.basicConfig(level=logging.INFO)                                                                                                                                       
logger = logging.getLogger("telegram-check")  

import socket                                                                                                                                                   
from urllib.parse import urlparse                                                                                                                                                                                                                                                                                              
from requests_toolbelt.adapters.host_header_ssl import HostHeaderSSLAdapter                                                                                                   
                                                                                                                                                                            
relay_url = (os.getenv("TELEGRAM_RELAY_URL") or "").strip()                                                                                                                   
host = urlparse(relay_url).hostname                                                                                                                                           
                                                                                                                                                                            
st.write("relay_url_set", bool(relay_url))                                                                                                                                    
st.write("relay_host", host)
                                                                                                                                                                            
if host:                                                                                                                                                                      
  try:                                                                                                                                                                      
      st.write("host_dns", socket.gethostbyname(host))                                                                                                                      
  except Exception as e:                                                                                                                                                    
      st.write("host_dns", f"DNS_FAIL: {e}")                                                                                                                                
                                                                                                                                                                            
try:                                                                                                                                                                          
  r = requests.get(relay_url, timeout=10)                                                                                                                                   
  st.write("relay_get_status", r.status_code)  # expect 405                                                                                                                 
except Exception as e:                                                                                                                                                        
  st.write("relay_get_error", str(e))
                                                                                                                                                                            
                                                                                                                                                                            
def _mask(value: str, keep: int = 4) -> str:                                                                                                                                  
  if not value:                                                                                                                                                             
      return "<missing>"                                                                                                                                                    
  if len(value) <= keep:                                                                                                                                                    
      return "*" * len(value)                                                                                                                                               
  return "*" * (len(value) - keep) + value[-keep:]                                                                                                                          
                                                                                                                                                                            
                                                                                                                                                                            
                                                                                                                                                                                
WORKER_HOST = "sweet-mountain-cc71.sufailsalim07.workers.dev"                                                                                                                 
WORKER_IPS = ["104.21.62.162", "172.67.137.22"]  # from your nslookup                                                                                                         
                                                                                                                                                                                
def _send_telegram_message(text: str) -> tuple[bool, str]:                                                                                                                    
  relay_key = os.getenv("TELEGRAM_RELAY_KEY")                                                                                                                               
  if not relay_key:                                                                                                                                                         
      return False, "Missing TELEGRAM_RELAY_KEY"                                                                                                                            
                                                                                                                                                                            
  s = requests.Session()                                                                                                                                                    
  s.mount("https://", HostHeaderSSLAdapter())                                                                                                                               
                                                                                                                                                                            
  for ip in WORKER_IPS:                                                                                                                                                     
      try:                                                                                                                                                                  
          r = s.post(                                                                                                                                                       
              f"https://{ip}/",                                                                                                                                             
              headers={                                                                                                                                                     
                  "Host": WORKER_HOST,                                                                                                                                      
                  "x-relay-key": relay_key,                                                                                                                                 
                  "content-type": "application/json",                                                                                                                       
              },                                                                                                                                                            
              json={"text": text},                                                                                                                                          
              timeout=15,                                                                                                                                                   
          )                                                                                                                                                                 
          if r.ok:                                                                                                                                                          
              return True, "Message sent via relay IP fallback"                                                                                                             
      except Exception:                                                                                                                                                     
          pass                                                                                                                                                              
                                                                                                                                                                            
  return False, "All relay IP attempts failed"                                                                                                                                      
                                                                                                                                                                            
                                                                                                                                                                            
@st.cache_resource                                                                                                                                                            
def startup_telegram_check() -> dict:                                                                                                                                         
  msg = f"HF startup check OK at {datetime.now(timezone.utc).isoformat()}"                                                                                                  
  ok, detail = _send_telegram_message(msg)                                                                                                                                  
  if ok:
      logger.info("Telegram startup check passed")                                                                                                                          
      return {"ok": True, "detail": detail}                                                                                                                                 
  logger.warning("Telegram startup check failed: %s", detail)                                                                                                               
  return {"ok": False, "detail": detail}                                                                                                                                    
                                                                                                                                                                            
                                                                                                                                                                            
st.set_page_config(page_title="Polymarket Bot Monitor", page_icon=":satellite:", layout="centered")                                                                           
st.title("Polymarket Bot Monitor")                                                                                                                                            
                                                                                                                                                                            
status = startup_telegram_check()                                                                                                                                             
st.caption(f"Startup Telegram check: {'OK' if status['ok'] else 'FAILED'}")                                                                                                   
if not status["ok"]:                                                                                                                                                          
  st.warning(status["detail"])                                                                                                                                              
                                                                                                                                                                            
relay_url = os.getenv("TELEGRAM_RELAY_URL")                                                                                                                                   
relay_key = os.getenv("TELEGRAM_RELAY_KEY")                                                                                                                                   
st.write(                                                                                                                                                                     
  f"Relay secret check: url={'set' if relay_url else 'missing'}, key={_mask(relay_key)}"                                                                                    
)                                                                                                                                                                             
                                                                                                                                                                            
if st.button("Send Telegram test message"):                                                                                                                                   
  ok, detail = _send_telegram_message(
      f"Manual test from HF at {datetime.now(timezone.utc).isoformat()}"                                                                                                    
  )                                                                                                                                                                         
  if ok:                                                                                                                                                                    
      st.success("Telegram test sent.")                                                                                                                                     
  else:                                                                                                                                                                     
      st.error(detail)