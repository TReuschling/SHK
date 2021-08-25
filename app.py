import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import scipy.special
import altair as alt
from io import BytesIO
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from PIL import Image
from vega_datasets import data
#import graphviz as graphviz

st. set_page_config(layout="wide")

nav = st.sidebar.radio("Navigation", [ "Server", "Bibliotheken"])
if nav == "Server":
    st.title("Server")
    '''
    Auf die Frage, wie das Projekt mittels der Bibliothek Streamlit gehostet werden kann, gibt es verschiedene Antworten.  
    Weiter muss man unterscheiden, zwischen dem Hosting für Streamlit und JupyterHub.  
    So wie ich es verstanden habe kann man Streamlit entweder einzel oder im Verbund mit einem JupyterHub hosten.  
      
    Wahrscheinlich ist es am sinnvollsten, wenn  (wie Dr.Yadav in der E-mail erwähnte) über einen Uni-Server, welcher professionell gewartet wird ein JupyterHub gehostet wird und
    auf diesem Hub dann Streamlit gehostet wird.  
      
    Cloud- / Serverbasierte Lösungen werden unten für den Hub aufgeführt.
    '''
    st.header("Streamlit-Intern")
    '''  
    Im Normalfall entnimmt Streamlit den Sourcode und die Anforderungen direkt aus dem angegebenen Github-repo.  
      
    1. Über einen E-mail-Account ist jeder Nutzer von Streamlit befähigt 3 Apps mit einer insgesamten max. Größe von 1GB hochzuladen.
    Je nach größe des Projekts könnten einfach mehrere Accounts eingerichtet werden.  
      
    2. Mit __Streamlit-Teams__ können unbegrenzt Apps hochgeladen werden. Weiter gibt es privates Repos, Single Sign-on (SSO) und eine sichere Verbindung zu privaten Daten.
    Allerdings ist __Streamlit-Teams__ noch in der Beta-Phase  
      
    https://streamlit.io/for-teams  
      
    Eine Anfrage wurde gestellt.
    '''
    st.header("Streamlit und JupyterHub")
    '''
    Mit den Dashborads von JupyterHub kann jeder Mitarbeiter Zugriff auf die Apps ähnlich wie bei einem gemeinsamen (privaten) Github-Repo haben.
    Die Authentifizierung wäre ähnlich wie bei Streamlit-Teams bzw. Jupyter-Hub über SSO oder Lightweight Directory Access Protocol (LDAP) möglich.

    Analog zu dem hosting via Github, können die Apps (deren Sourcecodes und Dependencies) auch in einem Dashboard hinterlegt werden und gemeinsam bearbeitet werden, bevor sie veröffentlich werden.

    Ausfürhliche Informationen unter:

    https://medium.com/@dslester/streamlit-apps-deploy-instantly-with-zero-configuration-7729944c649c
    https://cdsdashboards.readthedocs.io/en/stable/chapters/userguide/frameworks/streamlit.html
    '''
    st.header("Cloud / Plattform")
    '''
    Ich habe nicht ganz verstanden, wie viel Interaktion mit dem Hub von seiten der Nutzer vorhanden sein soll oder ob er nur für die interne Entwicklung da ist.  
    Hier eine kurze Aufzählung der potentiellen Möglichkeiten (mit Links zu den Preiskalkulatoren) für ein internes oder externen Hosting des JupyterHub:  
      
    __JupyterHub for Kubernetes - Self-Hosted:__  
      
    - am kompliziertesten
    - konfiguration dauert am längsten 
    - empfohlen für Gruppen zw. 100 - 1000 Benutzer  
    - open source  
    - gro0e community / guter support
           
    __The Littlest JupyterHub - Self-Hosted:__  
      
    - für kleinere Gruppen unter 50 Benutzer
    - wird auf einer virtuellen Maschine distributiert
    - gro0e community / guter support
      
    __Visual Studio Online - Hosted for You:__
      
    - cloud betrieben
    - kann als zentraller oder individueller Hub eingerichtet werden
    - für Gruppen unter 100 Benutzer
    - gut skalierbar und schnelle Konfiguration
      
    __Local Jupyter Notebooks - Self-Hosted:__
      
    - lokal auf eigener Hardware
    - kein Limit für Benutzer (limitiert durch die zugrunde liegende Hardware)
    - schwieriger eine standartisierte Umgebung für alle User zu schaffen
    - open source
    - große community / guter support
      
    Anmerkung zu dem Bild:  
      
    Spalte 1 (Colab, Binder...) ist nicht relevant, da es sich um ein JupyterNotebook geht.  
      
    '''
    image = Image.open("https://github.com/TReuschling/SHK/blob/main/Aufstellung1.png")
    st.image(image, caption = "Aufstellung")
    '''
    __Preiskalkulation für Littlest Jupyterhub:__  
      
    Folgende Annahmen werden liegen der Berechnung zu Grunde (welche auch ungefähr für Kubernetes gelten):  
      
    1. Recommended Memory = (Maximum Concurrent Users x Maximum Memory per User) + 128 MB  
    2. Recommended vCPUs = (Maximum Concurrent Users x Maximum CPU Usage per User) + 20%
    3. Recommended Disk Size = (Total Users x Maximum Disk Usage per User) + 2 GB  
    
    - Maximal 50% der Benutzer greifen gleichzeitig zu.

    '''
    image = Image.open("/Users/tassiloreuschling/Uni/SHK_Stelle/Code/Bilder/Aufstellung2.png")
    st.image(image, caption = "Aufstellung")
    '''
    __Preiskalkulation für JupyterHub mit Kubernetes:__  
      
    - Ungefähr 25% der Benutzer greifen gleichzeitig zu
    - Automatische Skalieren ist ein Feature von Kubernetes und hier für die Kostenreduktion verantwortlich
    -> Skaliert in der Nacht und am Wochenende runter und bei Bedarf hoch  

    https://www.scaleuptech.com/de/blog/autoskalierung-in-kubernetes/  

    '''
    image = Image.open("/Users/tassiloreuschling/Uni/SHK_Stelle/Code/Bilder/Aufstellung3.png")
    st.image(image, caption = "Aufstellung")
    '''
    __Links zu den Preisrechnern:__  
      

    Amazon Web Services (AWS)  
    Preise: https://aws.amazon.com/ec2/pricing/on-demand/  
      
    Azure (Microsoft)  
    Preise: https://azure.microsoft.com/en-us/pricing/calculator/  
      
    Google Cloud Platform (GCP)  
    Preise: https://cloud.google.com/compute/vm-instance-pricing  
      
    Digital Ocean  
    Preise: https://www.digitalocean.com/pricing/  

    '''

if nav == "Bibliotheken":

    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    c = alt.Chart(df).mark_circle().encode(
        x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

    # prepare some data
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 2, 4, 5]

    st.title("Liste von möglichen Bibliotheken")
    '''
    Wichtig für die Auswahl ist, dass sie open-source sind.
    '''
    st.header("1. Mathematisch - Bibliotheken")
    '''
    __NumPy__  
      
    NumPy ist eine Programmbibliothek für die Programmiersprache Python, die eine einfache Handhabung von Vektoren,
    Matrizen oder generell großen mehrdimensionalen Arrays ermöglicht.
    Neben den Datenstrukturen bietet NumPy auch effizient implementierte Funktionen für numerische Berechnungen an.  
      
    https://numpy.org/
    '''
    st.header("2. Datenmanagement - Bibliotheken")
    '''
    __Pandas__  
      
    Pandas ist eine Programmbibliothek für die Programmiersprache Python, die Hilfsmittel für die Verwaltung von Daten und deren Analyse anbietet.
    Insbesondere enthält sie Datenstrukturen und Operatoren für den Zugriff auf numerische Tabellen und Zeitreihen. pandas ist Freie Software,
    veröffentlicht unter der 3-Klausel-BSD-Lizenz.  
      
    Mit Hilfe von integrierten Streamlit-Plotting-Funktionn können Daten aus einem Pandas-Dataframe einfach veranschaulicht werden:  
    (Und das Bild kann sofort! ohne weitere Arbeit runtergeladen werden. Auch ein Knopf für das Runterladen der CSV Datei ist einfach zu implementieren)  

    https://pandas.pydata.org/
    '''
    st.write(" ")
    st.write(c)
    '''
    Weitere graphische Darstellungsmöglichkeite sind integriert oder mit externen Bibliothen möglich:
    '''
    st.header("3. Mathematische Plotting - Bibliotheken")
    '''
    __Streamlit__  
      
    Streamlit selbst bietet viele Möglichkeiten um einfache Graphiken zu implementieren, wie zb.:  
    - line_chart  
    '''
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)
    '''
    - area_chart  
    '''
    st.area_chart(chart_data)
    '''
    - bar_chart  
    '''
    st.bar_chart(chart_data)
    '''

    __Matplotlib__  
    - Matplotlib ist die Standwartwahl für graphische Dastellungen im wissenschaftlichen Bereiche
    - sehr resourceneffizient (wichtig für große Projekte)
    - eigentlich nicht sehr interaktiv, kann aber mithilfe der Streamlit - Bibiliothek interaktiv gestaltet werden
    '''
    t = np.arange(0., 5., 0.2)
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(5, 5))
    Exp = st.slider("Exponent für die gelbe Funktion", 0., 5., 2.5, 0.1)
    x = plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^', t, t**Exp, 'y')
    #contour = plt.contour(x, y, h)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    '''
      https://matplotlib.org/  
    '''
    st.image(buf)
    '''

    __Bookeh__  
    - "schönere" Darstellung
    - kann interaktiver gestaltet werden (aber aufwending in der Verbindung mit Streamlit)
    - kompliziert in der Implementierung
    '''
    # x = [1, 2, 3, 4, 5]
    # y = [6, 7, 2, 4, 5]

    # p = figure(
    # title='simple line example',
    # x_axis_label='x',
    # y_axis_label='y')

    # p.line(x, y, legend_label='Trend', line_width=2)

    # st.bokeh_chart(p, use_container_width=True)


    def make_plot(title, hist, edges, x, pdf, cdf):
        p = figure(title=title, tools='', background_fill_color="#fafafa")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="navy", line_color="white", alpha=0.5)
        p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
        p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")

        p.y_range.start = 0
        p.legend.location = "center_right"
        p.legend.background_fill_color = "#fefefe"
        p.xaxis.axis_label = 'x'
        p.yaxis.axis_label = 'Pr(x)'
        p.grid.grid_line_color="white"
        return p

    # Normal Distribution

    mu, sigma = 0, 0.5

    measured = np.random.normal(mu, sigma, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)

    x = np.linspace(-2, 2, 1000)
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

    p1 = make_plot("Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)

    # Log-Normal Distribution

    mu, sigma = 0, 0.5

    measured = np.random.lognormal(mu, sigma, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)

    x = np.linspace(0.0001, 8.0, 1000)
    pdf = 1/(x* sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))
    cdf = (1+scipy.special.erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))/2

    p2 = make_plot("Log Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)

    # Gamma Distribution

    k, theta = 7.5, 1.0

    measured = np.random.gamma(k, theta, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)

    x = np.linspace(0.0001, 20.0, 1000)
    pdf = x**(k-1) * np.exp(-x/theta) / (theta**k * scipy.special.gamma(k))
    cdf = scipy.special.gammainc(k, x/theta)

    p3 = make_plot("Gamma Distribution (k=7.5, θ=1)", hist, edges, x, pdf, cdf)

    # Weibull Distribution

    lam, k = 1, 1.25
    measured = lam*(-np.log(np.random.uniform(0, 1, 1000)))**(1/k)
    hist, edges = np.histogram(measured, density=True, bins=50)

    x = np.linspace(0.0001, 8, 1000)
    pdf = (k/lam)*(x/lam)**(k-1) * np.exp(-(x/lam)**k)
    cdf = 1 - np.exp(-(x/lam)**k)

    p4 = make_plot("Weibull Distribution (λ=1, k=1.25)", hist, edges, x, pdf, cdf)

    output_file('histogram.html', title="histogram.py example")

    st.bokeh_chart(gridplot([p1,p2,p3,p4], ncols=2, plot_width=400, plot_height=400, toolbar_location=None))
    '''

    https://bokeh.org/  
      
    '''
    
    '''
    __Altair__  
      
    - komplexe / "schöne" Darstellung bei wenig Aufwand  
    - gut für die Darstellung von CSV - Dateien bzw Pandas-Dataframes  
      
    BeispielPlots:  

    '''
    iris = data.iris()

    ax = alt.Chart(iris).mark_point().encode(
        x='petalLength',
        y='petalWidth',
        color='species'
    )
    st.altair_chart(ax, use_container_width=True)
    #st.altair_chart(c, use_container_width=True)

    states = alt.topo_feature(data.us_10m.url, feature='states')
    airports = data.airports.url

    # US states background
    background = alt.Chart(states).mark_geoshape(
        fill='lightgray',
        stroke='white',
        ).properties(
        width=800,
        height=500
        ).project('albersUsa')

    # airport positions on background
    points = alt.Chart(airports).mark_circle().encode(
        longitude='longitude:Q',
        latitude='latitude:Q',
        size=alt.value(15),
        color=alt.value('#3377B3'),
        tooltip=['iata:N','name:N','city:N','state:N','latitude:Q','longitude:Q'],
        )

    chart = (background + points)
    chart.save('airports.html')
    st.altair_chart(chart, use_container_width=True)
    '''
      
    https://altair-viz.github.io/
    '''
    st.header("4. Illustrierende Plotting - Bibliotheken")
    '''
    __Graphviz__  
      
    - Gut für die Darstellung von Zusammenhängen in Form eines Diagramms / Netzwerks
    - Ich hatte Probleme es die Pakete für das Graphik - Rendern zum laufen zu bringen. Sie schreiben aber auch, dass sie für OSX noch Hilfe brauchen für das richtige Installationspaket.  
      
    https://www.graphviz.org/
    '''
    image = Image.open("/Users/tassiloreuschling/Uni/SHK_Stelle/Code/Bilder/Graphviz.png")
    st.image(image)
    '''
    __Pyvis__  
      
    - Netzwerkdarstellung
    - Alternative zu Graphivz
    - Eher Interaktiv
      
    https://pyvis.readthedocs.io/en/latest/  
    https://discuss.streamlit.io/t/interactive-networks-graphs-with-pyvis/8344
    '''
    st.header("5. Weitere Bibliotheken")
    '''
    __Pytorch__ (Maschinelles Lernen)  
      
    PyTorch ist eine auf Maschinelles Lernen ausgerichtete Open-Source-Programmbibliothek für die Programmiersprache Python,
    basierend auf der in Lua geschriebenen Bibliothek Torch, die bereits seit 2002 existiert.  
      
    https://pytorch.org/  

    __Scikit - learn__ (Maschinelles Lernen)
      
    Scikit-learn (ehemals scikits.learn) ist eine freie Software-Bibliothek zum maschinellen Lernen für die Programmiersprache Python.
    Es bietet verschiedene Klassifikations-, Regressions- und Clustering-Algorithmen, darunter Support-Vektor-Maschinen, Random Forest,
    Gradient Boosting, k-means und DBSCAN. Sie basiert als SciKit (Kurzform für SciPy Toolkit), wie beispielsweise auch Scikit-image,
    auf den numerischen und wissenschaftlichen Python-Bibliotheken NumPy und SciPy.  
      
    https://github.com/scikit-learn/scikit-learn
    '''
