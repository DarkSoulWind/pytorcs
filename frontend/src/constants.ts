const isDev = true;
const DEV_API = "http://localhost:8000";
const PROD_API = "www.buss.com";
export const API_URL = isDev ? DEV_API : PROD_API;
