# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### Configuring Credentials

The app expects user credentials as a mapping of usernames to **SHA-256 hashed**
passwords. Provide them either via the environment variable `NUMERONIQ_CREDENTIALS`
or in a `credentials.json` file in the project root (ignored by Git). Example:

```json
{
  "admin": "<hashed-password>",
  "user": "<hashed-password>"
}
```
