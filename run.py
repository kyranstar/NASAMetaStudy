from simulationgui.app import app

if __name__ == '__main__':
    # Dash CSS
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
    # Loading screen CSS
    # app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
    app.run_server(debug=True)
