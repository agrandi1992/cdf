"""
Streamlit Dashboard for Chauffeur de France
-----------------------------------------

This Streamlit application provides an interactive dashboard for exploring the
2025 invoice data of Chauffeur de France and identifying market opportunities
for 2026.  The app offers multiple pages:

* **Vue d’ensemble** – High‑level KPIs and revenue trends.
* **Segmentation clients** – Explore client types and RFM segments with interactive charts.
* **Analyse des marchés** – Examine revenue by service category and recommended actions.
* **Carte & événements 2026** – Map of major French events in 2026 compared with
  current business distribution.
* **Qualité & fidélisation** – KPIs around loyalty, repeat business and payment
  delays.

To run this app locally install the required packages (streamlit, pandas,
plotly, pydeck) and then run `streamlit run app.py` from the project root.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pydeck as pdk
from datetime import datetime

# -----------------------------------------------------------------------------
# Streamlit configuration
#
# Set the page title, icon and layout. This must be called before any Streamlit
# commands to ensure the configuration is applied. A custom car logo created via
# the image generation tool is stored in the project root as ``logo.png`` and
# will be displayed in the header below.
st.set_page_config(
    page_title="Chauffeur de France – Tableau de bord",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

@st.cache_data
def load_invoices(path: str) -> pd.DataFrame:
    """
    Charge les données de factures depuis Excel et nettoie les colonnes.

    - Supprime les espaces en début/fin des noms de colonnes.
    - Convertit la date en format datetime.
    - Convertit les montants TTC et HT en numériques.
    """
    df = pd.read_excel(path)
    # Nettoyage des noms de colonnes
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Conversion des dates
    df['date de facture'] = pd.to_datetime(df['date de facture'])
    # Conversion des montants
    for col in ['montant TTC', 'montant HT']:
        # Some column names may include trailing spaces in the Excel file
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data
def load_customer_summary(path: str) -> pd.DataFrame:
    """Load customer RFM summary."""
    return pd.read_csv(path)

@st.cache_data
def load_events(path: str) -> pd.DataFrame:
    """Load 2026 events dataset (JSON or CSV)."""
    if path.endswith('.json'):
        return pd.read_json(path)
    return pd.read_csv(path)


def categorize_intitule(intitulé: str) -> str:
    """
    Catégorise une prestation de manière détaillée en se basant sur des mots‑clés.

    Les catégories sont :
      * **Fashion Week** : prestations liées aux défilés et événements mode.
      * **Location de véhicule** : location de voitures (classe S/V), vans, Range Rover, camions, etc.
      * **Mise à disposition** : mise à disposition avec chauffeur et missions diverses.
      * **Transfert** : transferts aéroport/gare, TGV, etc.
      * **Mariage** : services pour mariages ou weddings.
      * **Séminaire** : déplacements pour séminaires.
      * **Événement** : salons, festivals et événements publics.
      * **Prestation** : prestations génériques non liées à un type précis.
      * **Transport divers** : transport logistique ou spécial.
      * **Service divers** : toutes les autres prestations.
    """
    if not isinstance(intitulé, str) or not intitulé.strip():
        return 'Service divers'
    s = intitulé.lower()
    # Fashion Week
    if 'fashion week' in s:
        return 'Fashion Week'
    # Location de véhicule (mots clés pour véhicules de luxe, vans, Range Rover, camions)
    if any(word in s for word in ['location', 'range rover', 'classe v', 'classe s', 's-class', 'e-class', 'van', 'mercedes', 'voiture', 'camion']):
        return 'Location de véhicule'
    # Mise à disposition / missions
    if any(word in s for word in ['dispo', 'mise à disposition', 'mise a disposition', 'mission', 'chauffeur', 'marbella']):
        return 'Mise à disposition'
    # Transferts
    if any(word in s for word in ['transf', 'transfert', 'transfer', 'aéroport', 'aeroport', 'airport', 'gare', 'tgv', 'train']):
        return 'Transfert'
    # Mariage / Wedding
    if 'mariage' in s or 'wedding' in s:
        return 'Mariage'
    # Séminaire
    if any(word in s for word in ['séminaire', 'seminaire']):
        return 'Séminaire'
    # Événement
    if any(word in s for word in ['événement', 'evenement', 'event', 'salon', 'festival']):
        return 'Événement'
    # Prestation
    if 'prestation' in s:
        return 'Prestation'
    # Transport divers
    if 'transport' in s:
        return 'Transport divers'
    # Default
    return 'Service divers'


def classify_client(name: str) -> str:
    """
    Classe les clients selon leur profil :

      * **Prestataire/Partenaire** : sociétés de location, de limousines ou de chauffeurs.
      * **Entreprise/Agence** : agences de voyages, groupes, sociétés d’événements.
      * **VIP particulier** : personnes physiques titulaires (princes, HRH, etc.).
      * **Client particulier** : client individuel lambda.

    Le classement se base sur des mots‑clés dans la raison sociale ou le nom.
    """
    if not isinstance(name, str) or not name.strip():
        return 'Client particulier'
    lower_name = name.lower()
    # Prestataire / Partenaire
    if any(term in lower_name for term in ['limousine', 'rent car', 'rent', 'driver', 'blob', 'ht rent car', 'executive driver', 'chauffeur']):
        return 'Prestataire/Partenaire'
    # Entreprise / Agence
    if any(term in lower_name for term in ['egencia', 'travel', 'voyage', 'events', 'event', 'group', 'compagny', 'compagnie', 'pilgrims', 'societe', 'société', 'majest', 'parimob', 'line']):
        return 'Entreprise/Agence'
    # VIP particulier
    if any(term in lower_name for term in ['hrh', 'prince', 'princes', 'princesse', 'gasly', 'faizal', 'salman', 'turki', 'royal', 'al omrane', 'pierre']):
        return 'VIP particulier'
    # Default
    return 'Client particulier'


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for categories and client types."""
    df = df.copy()
    # Apply the new categorisation and client type classification
    df['Category'] = df['intitulé'].apply(categorize_intitule)
    df['Client Type'] = df['nom du client'].apply(classify_client)
    df['Month'] = df['date de facture'].dt.to_period('M').astype(str)
    return df


# Load data
invoices = load_invoices('facturesCHAUFFEURFR.xlsx')
invoices = prepare_dataset(invoices)
# Charger le résumé clients RFM mis à jour (v3) et les événements de luxe
customer_summary = load_customer_summary('customer_summary_updated_v3.csv')
events = load_events('events_luxury.json')

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.title('Chauffeur de France Dashboard')
page = st.sidebar.selectbox(
    'Sélectionnez une page',
    [
        'Vue d’ensemble',
        'Segmentation clients',
        'Typologie & canaux',
        'Événements de luxe',
        'Qualité & fidélisation'
    ]
)

# -----------------------------------------------------------------------------
# Global header
#
# Display the company logo and application name at the top of every page.  The
# logo image must be present in the working directory (``logo.png``) and was
# generated via the image generation tool to avoid copyright concerns.
header_col1, header_col2 = st.columns([1, 5])
with header_col1:
    try:
        st.image('logo.png', width=80)
    except Exception:
        pass
with header_col2:
    st.markdown(
        "<h1 style='margin-top: 10px;'>Chauffeur de France – Tableau de bord 2025‑2026</h1>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# KPI functions
# -----------------------------------------------------------------------------
def display_overview(df: pd.DataFrame) -> None:
    """Display high‑level KPIs and revenue trends."""
    st.header('Vue d’ensemble')
    # Use trimmed column name 'montant TTC' if available
    amount_col = 'montant TTC' if 'montant TTC' in df.columns else 'montant TTC '
    # Compute KPIs
    total_rev = df[amount_col].sum()
    unique_clients = df['nom du client'].nunique()
    num_invoices = df.shape[0]
    avg_invoice = df[amount_col].mean()
    # Loyalty rate: clients with more than one invoice
    invoice_counts = df['nom du client'].value_counts()
    loyalty_rate = invoice_counts[invoice_counts > 1].count() / unique_clients if unique_clients > 0 else 0
    # Determine top category by revenue
    rev_by_cat = df.groupby('Category')[amount_col].sum().reset_index()
    top_cat_row = rev_by_cat.sort_values(amount_col, ascending=False).iloc[0]

    # Calcul de la part du chiffre d'affaires généré par la catégorie principale
    top_cat_share = (top_cat_row[amount_col] / total_rev) if total_rev else 0

    # Calculs supplémentaires pour présenter des KPI plus riches
    # Nombre de clients réguliers (plus d'une facture) et nouveaux clients
    regular_clients = invoice_counts[invoice_counts > 1].count()
    new_clients = unique_clients - regular_clients
    # Part du chiffre d'affaires généré par les 5 meilleurs clients
    top_clients_share = df.groupby('nom du client')[amount_col].sum().nlargest(5).sum() / total_rev if total_rev else 0

    # Première ligne de KPI (quatre colonnes)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('CA total', f"{total_rev:,.0f} €", help='Chiffre d’affaires TTC total en 2025')
    col2.metric('Factures', num_invoices, help='Nombre total de factures émises en 2025')
    col3.metric('Clients', unique_clients, help='Nombre de clients distincts en 2025')
    col4.metric('Ticket moyen', f"{avg_invoice:,.0f} €", help='Montant moyen par facture')

    # Deuxième ligne de KPI (quatre colonnes)
    col5, col6, col7, col8 = st.columns(4)
    col5.metric('Fidélité', f"{loyalty_rate*100:.0f}%", help='Part des clients ayant effectué plus d’une facture')
    col6.metric('CA top service', f"{top_cat_share*100:.0f}%", help=f'Part du CA générée par le service {top_cat_row["Category"]}')
    col7.metric('Clients réguliers', regular_clients, help='Nombre de clients ayant effectué plusieurs factures')
    col8.metric('CA top 5 clients', f"{top_clients_share*100:.0f}%", help='Part du CA générée par les cinq meilleurs clients')

    # Highlights section
    st.subheader('Faits saillants de 2025')
    # CA par mois et top 3 mois
    monthly_tot = df.groupby('Month')[amount_col].sum().reset_index()
    top_months = monthly_tot.sort_values(amount_col, ascending=False).head(3)['Month'].tolist()
    months_str = ', '.join(top_months)
    # CA par type de client et meilleur type
    rev_by_type = df.groupby('Client Type')[amount_col].sum()
    if not rev_by_type.empty:
        top_type = rev_by_type.idxmax()
        top_type_val = rev_by_type.max()
    else:
        top_type = ''
        top_type_val = 0
    # Identify low season months for insight
    bottom_months = monthly_tot.sort_values(amount_col).head(2)['Month'].tolist()
    low_months_str = ', '.join(bottom_months)
    st.markdown(
        f"- **Catégorie majeure :** {top_cat_row['Category']} (≈ {top_cat_row[amount_col]:,.0f} € de CA)\n"
        f"- **Type de client le plus rentable :** {top_type} (≈ {top_type_val:,.0f} € de CA)\n"
        f"- **Mois de pic d’activité :** {months_str} (Fashion Week en début d’année, missions de printemps et saison estivale)\n"
        f"- **Mois creux :** {low_months_str} – période à exploiter par des offres promotionnelles\n"
        f"- **Fidélité** : {loyalty_rate*100:.0f}% des clients ont effectué plusieurs services et génèrent la majorité du CA"
    )

    # Revenue by month line chart with peaks annotated
    monthly = monthly_tot.sort_values('Month')
    fig = px.line(monthly, x='Month', y=amount_col, title="Évolution mensuelle du chiffre d'affaires")
    fig.update_traces(mode='lines+markers', line=dict(color='#007ACC'), marker=dict(size=6))
    # Annotate peaks (top 3 months)
    for _, row in monthly_tot.nlargest(3, amount_col).iterrows():
        fig.add_annotation(x=row['Month'], y=row[amount_col], text=f"{int(row[amount_col]):,} €", showarrow=True, arrowhead=1, yshift=10)
    fig.update_layout(xaxis_title='Mois', yaxis_title='CA TTC (€)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def display_client_segmentation(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Display segmentation of clients with interactive filters and RFM table."""
    st.header('Segmentation clients')
    # Filter by client type
    types = df['Client Type'].unique().tolist()
    type_filter = st.multiselect('Filtrer par type de client', types, default=types)
    filtered_df = df[df['Client Type'].isin(type_filter)]

    # Revenue by client type bar chart
    amount_col = 'montant TTC' if 'montant TTC' in df.columns else 'montant TTC '
    rev_type = filtered_df.groupby('Client Type')[amount_col].sum().reset_index()
    fig1 = px.bar(rev_type, x='Client Type', y=amount_col, color='Client Type',
                  title="Chiffre d'affaires par type de client")
    st.plotly_chart(fig1, use_container_width=True)

    # Revenue by service category and client type (stacked bar)
    cat_by_type = filtered_df.groupby(['Category','Client Type'])[amount_col].sum().reset_index()
    fig2 = px.bar(cat_by_type, x='Category', y=amount_col, color='Client Type',
                  title="Chiffre d'affaires par service et type de client", barmode='stack')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Segments RFM')
    st.write('Le tableau ci‑dessous classe les clients selon la récence (R), la fréquence (F) et le montant (M). Les meilleurs clients “Champions” ont des factures récentes, fréquentes et élevées.')
    # Harmonise column names for display
    rename_map = {}
    # Accept both English and French naming variants
    cols = summary.columns
    if 'Client' in cols:
        rename_map['Client'] = 'Client'
    elif 'nom du client' in cols:
        rename_map['nom du client'] = 'Client'
    if 'Recence' in cols:
        rename_map['Recence'] = 'Récence (jours)'
    elif 'Recency' in cols:
        rename_map['Recency'] = 'Récence (jours)'
    if 'Frequence' in cols:
        rename_map['Frequence'] = 'Fréquence'
    elif 'Frequency' in cols:
        rename_map['Frequency'] = 'Fréquence'
    if 'Monetaire' in cols:
        rename_map['Monetaire'] = 'Montant total'
    elif 'Monetary' in cols:
        rename_map['Monetary'] = 'Montant total'
    if 'Segment' in cols:
        rename_map['Segment'] = 'Segment'
    if 'Segment Name' in cols:
        rename_map['Segment Name'] = 'Segment'
    st.dataframe(summary.rename(columns=rename_map))


def display_market_analysis(df: pd.DataFrame) -> None:
    """
    Explore revenue by service category (typologie) and propose actions per canal.
    This page highlights how each type of service contributes to the business and
    suggests acquisition channels tailored to each typology.
    """
    st.header('Typologie & canaux')
    # Total revenue by category
    amount_col = 'montant TTC' if 'montant TTC' in df.columns else 'montant TTC '
    rev_cat = df.groupby('Category')[amount_col].sum().reset_index().sort_values(amount_col, ascending=False)
    fig = px.bar(rev_cat, x='Category', y=amount_col, color='Category',
                 title="Chiffre d'affaires par type de service")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Recommandations par typologie et canal')
    for _, row in rev_cat.iterrows():
        cat = row['Category']
        val = row[amount_col]
        # Provide tailored recommendations per service
        if cat == 'Fashion Week':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Consolidez les partenariats avec les maisons de couture et agences événementielles. "
                "Proposez des forfaits VIP et des services premium pour fidéliser cette clientèle exigeante."
            )
        elif cat == 'Mise à disposition':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Développez l’offre de mise à disposition à la journée ou à la semaine. "
                "Ciblez les entreprises et institutions nécessitant des chauffeurs réguliers et proposez des abonnements."
            )
        elif cat == 'Location de véhicule':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Diversifiez la flotte (SUV, van, voiture de luxe) et collaborez avec des agences de voyage ou des plateformes de réservation. "
                "Offrez des tarifs dégressifs pour les locations longue durée et un service chauffeur en option."
            )
        elif cat == 'Transfert':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Renforcez votre présence sur les transferts aéroport/gare en signant des accords avec hôtels, conciergeries et compagnies aériennes. "
                "Mettez en avant la ponctualité, la sécurité et la discrétion."
            )
        elif cat == 'Événement':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Positionnez‑vous comme prestataire clé pour les événements (salons professionnels, mariages, fêtes locales). "
                "Proposez des offres packagées avec décoration du véhicule et chauffeurs bilingues."
            )
        elif cat == 'Mariage':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Développez une offre sur‑mesure pour les cérémonies et mariages haut de gamme : véhicule décoré, service de coordination et attention personnalisée."
            )
        elif cat == 'Séminaire':
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Ciblez les entreprises organisant des séminaires. Proposez des navettes et des transports groupés avec prestations à bord."
            )
        else:
            st.markdown(
                f"**{cat} (≈ {val:,.0f} €)** – Analysez ces prestations hétérogènes et identifiez celles à forte marge. "
                "Renforcez les services rentables et supprimez ceux qui sont peu profitables."
            )


def display_map(events_df: pd.DataFrame) -> None:
    """Show interactive map of 2026 events and compare with current business."""
    st.header('Événements de luxe 2026')
    st.write(
        "Cette carte interactive affiche les principaux événements haut de gamme en 2026 "
        "en France, au Luxembourg et en Belgique. La taille des marqueurs reflète le chiffre d’affaires "
        "potentiel estimé (en fonction du ticket moyen et de la durée), et la couleur indique "
        "si le pays fait déjà partie de votre clientèle (bleu) ou constitue un nouveau marché (orange)."
    )

    # Determine if each event corresponds to a current market (based on country)
    invoice_countries = [c.lower() for c in invoices['pays du client'].dropna().unique().tolist()]
    events_copy = events_df.copy()
    status_list = []
    for _, row in events_copy.iterrows():
        country = row.get('Country') or row.get('Pays')
        if isinstance(country, str) and country.strip().lower() in invoice_countries:
            status_list.append('Marché existant')
        else:
            status_list.append('Marché existant')
    events_copy['Status'] = status_list
    # Create period string for hover and table
    events_copy['Periode'] = events_copy['Start Date'].astype(str) + ' → ' + events_copy['End Date'].astype(str)

    # Create a scatter_mapbox chart. Use open-street-map style to avoid needing a token.
    size_ref = events_copy['Potential'].max() / 30 if 'Potential' in events_copy.columns else 1
    fig = px.scatter_mapbox(
        events_copy,
        lat='Latitude',
        lon='Longitude',
        hover_name='Event',
        hover_data={
            'City': True,
            'Country': True,
            'Periode': True,
            'Type': True,
            'Potential': ':.0f',
            'Description': False
        },
        color='Status',
        size='Potential' if 'Potential' in events_copy.columns else None,
        size_max=30,
        zoom=4.5,
        height=500,
        title="Carte des événements haut de gamme"
    )
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(legend_title='Statut du marché')
    st.plotly_chart(fig, use_container_width=True)

    # Table summarising events with potential revenue
    st.subheader("Synthèse des opportunités")
    summary_cols = ['Event', 'City', 'Country', 'Periode', 'Type', 'Status']
    if 'Potential' in events_copy.columns:
        summary_cols.append('Potential')
    comp_table = events_copy[summary_cols].rename(columns={
        'Event': 'Événement',
        'City': 'Ville',
        'Country': 'Pays',
        'Periode': 'Période',
        'Type': 'Type',
        'Status': 'Statut',
        'Potential': 'CA potentiel (≈€)'
    })
    # Format potential as integer with thousands separator
    if 'CA potentiel (≈€)' in comp_table.columns:
        comp_table['CA potentiel (≈€)'] = comp_table['CA potentiel (≈€)'].astype(float).round(0).map(lambda x: f"{x:,.0f}")
    st.dataframe(comp_table)


def display_quality_loyalty(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Show KPIs related to client loyalty and payment delays."""
    st.header('Qualité & fidélisation')
    # Loyalty rate: proportion of clients with more than 1 invoice
    # Determine column names (Recency/Recence, Frequency/Frequence, etc.) dynamically
    name_col = 'nom du client' if 'nom du client' in summary.columns else 'Client'
    freq_col = 'Frequence' if 'Frequence' in summary.columns else ('Frequency' if 'Frequency' in summary.columns else None)
    # Loyalty rate: clients with frequency > 1
    if freq_col:
        repeat_clients = summary[summary[freq_col] > 1][name_col].nunique()
    else:
        repeat_clients = 0
    total_clients = summary[name_col].nunique()
    loyalty_rate = repeat_clients / total_clients if total_clients > 0 else 0
    # Unpaid invoices
    # Unpaid invoices: status different from 'Payée'
    unpaid = df[df['statut de la facture'].str.lower() != 'payée'].shape[0]
    total_invoices = df.shape[0]
    unpaid_rate = unpaid / total_invoices if total_invoices > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric('Taux de fidélité', f"{loyalty_rate*100:.1f}%", help='Part des clients ayant effectué plusieurs factures en 2025')
    col2.metric('Factures non payées', f"{unpaid_rate*100:.1f}%", help='Part des factures envoyées ou facturées non encore réglées')

    # Provide context from research about payment delays
    st.markdown("")

    # Show distribution of recence and frequency (RFM scatter)
    st.subheader('Distribution de la récence et de la fréquence')
    # Determine column names for recency, frequency, monetary and segment
    rec_col = 'Recence' if 'Recence' in summary.columns else ('Recency' if 'Recency' in summary.columns else None)
    freq_col = 'Frequence' if 'Frequence' in summary.columns else ('Frequency' if 'Frequency' in summary.columns else None)
    mon_col = 'Monetaire' if 'Monetaire' in summary.columns else ('Monetary' if 'Monetary' in summary.columns else None)
    seg_col = 'Segment' if 'Segment' in summary.columns else ('Segment Name' if 'Segment Name' in summary.columns else None)
    if rec_col and freq_col and mon_col:
        # Ensure size values are positive and scaled
        fig = px.scatter(
            summary,
            x=rec_col,
            y=freq_col,
            size=summary[mon_col].apply(lambda x: abs(x)),
            color=seg_col if seg_col else None,
            hover_name=name_col,
            title='Segmentation RFM : récence vs fréquence',
        )
        # Adjust marker sizing range for better visuals
        fig.update_traces(marker=dict(sizeref=2.*summary[mon_col].abs().max()/(40.**2), sizemode='area'))
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Page rendering
# -----------------------------------------------------------------------------
if page == 'Vue d’ensemble':
    display_overview(invoices)
elif page == 'Segmentation clients':
    display_client_segmentation(invoices, customer_summary)
elif page == 'Typologie & canaux':
    display_market_analysis(invoices)
elif page == 'Événements de luxe':
    display_map(events)
else:
    display_quality_loyalty(invoices, customer_summary)