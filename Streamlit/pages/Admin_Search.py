

import streamlit as st
import pandas as pd
import numpy as np
# import ssl
import os
import certifi
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
# import pgeocode
import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client, LocalCluster

os.environ['SSL_CERT_FILE'] = certifi.where()

# data = pd.read_csv('/Users/savethebees/Downloads/df_broad.csv')
# data['NTEE_CD'] = data['NTEE_CD'].str[:3]
# ntee_code = data['NTEE_CD']

def main():
# Example: Create a Dask client with 4 workers, each using 2 threads
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)


    meta = {
        'EIN': 'int64',
        'NAME': 'object',
        'Category': 'object',
        'ICO': 'object',
        'STREET': 'object',
        'CITY': 'object',
        'STATE': 'object',
        'ZIP': 'object',
        'lat': 'float64',
        'lon': 'float64',
        'PhoneNum_990': 'object',
        'ReturnTs_990': 'object',
        'PersonNm_990': 'object',
        'PersonTitleTxt_990': 'object',
        'WebsiteAddressTxt_990': 'object',
        'RecipientEmailAddressTxt_990': 'object',
        'ActivityOrMissionDesc_990': 'object',
        'MissionDesc_990': 'object',
        'PrimaryExemptPurposeTxt_990': 'object',
        'Desc_990': 'object',
        'DescriptionProgramSrvcAccomTxt_990': 'object',
        'Description1Txt_990': 'object'
    }


# # Load CSV as Dask DataFrame
#     data = dd.read_csv('/Users/savethebees/Downloads/df_broad.csv', dtype=meta)
    data = dd.read_csv('../Data/df_broad.csv', dtype=meta)

# Define NTEE categories
    ntee_categories = {
        'A': 'Arts, Culture and Humanities',
        'B': 'Education',
        'C': 'Environment',
        'D': 'Animal-Related',
        'E': 'Health Care',
        'F': 'Mental Health & Crisis Intervention',
        'G': 'Voluntary Health Associations & Medical Disciplines',
        'H': 'Medical Research',
        'I': 'Crime & Legal-Related',
        'J': 'Employment',
        'K': 'Food, Agriculture and Nutrition',
        'L': 'Housing & Shelter',
        'M': 'Public Safety, Disaster Preparedness and Relief',
        'N': 'Recreation & Sports',
        'O': 'Youth Development',
        'P': 'Human Services',
        'Q': 'International, Foreign Affairs and National Security',
        'R': 'Civil Rights, Social Action & Advocacy',
        'S': 'Community Improvement & Capacity Building',
        'T': 'Philanthropy, Voluntarism and Grantmaking Foundations',
        'U': 'Science & Technology',
        'V': 'Social Science',
        'W': 'Public & Societal Benefit',
        'X': 'Religion-Related',
        'Y': 'Mutual & Membership Benefit',
        'Z': 'Unknown'
    }

# Streamlit UI
    st.title("Admin Search")

# Main category selection
    main_category = st.radio("Choose a Category:", list(ntee_categories.values()))

# Find the corresponding code for the selected main category
    selected_code = [code for code, desc in ntee_categories.items() if desc == main_category][0]

# User inputs
    zip_code = st.text_input("Enter Zip Code")
    radius = st.number_input("Enter Search Radius (in miles)", min_value=1, max_value=101, value=10)

#Haversine distance function
    def haversine(lat1, lon1, lat2, lon2):
        R = 3958.8  # Radius of Earth in miles
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) #map() ?
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

# Geolocate user based on zip code
    if zip_code and radius:
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode(zip_code, timeout=10, country_codes='US')
        if location:
            user_lat, user_lon = location.latitude, location.longitude
            st.write("Location found:", (user_lat, user_lon))

        # Filter data by category and drop rows with NaN in lat/lon columns
            filtered_data = data[data['Category'] == selected_code]
            filtered_data = filtered_data.dropna(subset=['lat', 'lon'])

            filtered_data['lat'] = filtered_data['lat'].astype(float)
            filtered_data['lon'] = filtered_data['lon'].astype(float)

        # Calculate distances in parallel using Dask
            filtered_data['distance'] = filtered_data.map_partitions(
                lambda df: haversine(user_lat, user_lon, df['lat'], df['lon']),
                meta=('distance', 'float64')
            )
        

        # Filter by radius and compute the result
            within_radius = filtered_data[filtered_data['distance'] <= radius].compute()

        # Display results in Streamlit
            if not within_radius.empty:
                st.write("Filtered Organizations within radius:", within_radius)
            else:
                st.write("No organizations found within the specified radius.")
        else:
            st.write("Zip code not found.")

if __name__ == '__main__':
    main()






# # Convert a Pandas DataFrame to Dask DataFrame
# data = data = dd.read_csv('/Users/savethebees/Downloads/df_broad.csv', dtype = meta)  # Adjust the number of partitions as needed

# ntee_categories = {
#     'A': 'Arts, Culture and Humanities',
#     'B': 'Education',
#     'C': 'Environment',
#     'D': 'Animal-Related',
#     'E': 'Health Care',
#     'F': 'Mental Health & Crisis Intervention',
#     'G': 'Voluntary Health Associations & Medical Disciplines',
#     'H': 'Medical Research',
#     'I': 'Crime & Legal-Related',
#     'J': 'Employment',
#     'K': 'Food, Agriculture and Nutrition',
#     'L': 'Housing & Shelter',
#     'M': 'Public Safety, Disaster Preparedness and Relief',
#     'N': 'Recreation & Sports',
#     'O': 'Youth Development',
#     'P': 'Human Services',
#     'Q': 'International, Foreign Affairs and National Security',
#     'R': 'Civil Rights, Social Action & Advocacy',
#     'S': 'Community Improvement & Capacity Building',
#     'T': 'Philanthropy, Voluntarism and Grantmaking Foundations',
#     'U': 'Science & Technology',
#     'V': 'Social Science',
#     'W': 'Public & Societal Benefit',
#     'X': 'Religion-Related',
#     'Y': 'Mutual & Membership Benefit',
#     'Z': 'Unknown'
# }

# st.title("Admin Search")

# geolocator = Nominatim(user_agent="geoapi")



# # Main category selection with radio buttons
# main_category = st.radio("Choose a Category:", list(ntee_categories.values()))

# # Find the corresponding code for the selected main category
# selected_code = [code for code, desc in ntee_categories.items() if desc == main_category][0]

# # User inputs for location and radius
# zip_code = st.text_input("Enter Zip Code")
# radius = st.number_input("Enter Search Radius (in miles)", min_value=1, max_value=101, value=10)


# def haversine(lat1, lon1, lat2, lon2):
#     R = 3958.8  # Radius of Earth in miles
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     return R * c

# # Geolocate user based on zip code
# if zip_code and radius:
#     geolocator = Nominatim(user_agent="geoapi")
#     location = geolocator.geocode(zip_code, timeout=10, country_codes='US')
#     if location:
#         user_lat, user_lon = location.latitude, location.longitude
#         st.write("Location found:", (user_lat, user_lon))

#         # Filter data by category and drop rows with NaN in lat/lon columns
#         filtered_data = data[data['Category'].isin([selected_code])]
#         filtered_data = filtered_data.dropna(subset=['lat', 'lon'])

#         # Calculate distances in parallel using Dask
#         filtered_data['distance'] = delayed(haversine)(user_lat, user_lon, filtered_data['lat'], filtered_data['lon'])

#         # Filter by radius and compute the result
#         within_radius = filtered_data[filtered_data['distance'] <= radius].compute()  # Compute the final filtered result

#         # Display results in Streamlit
#         if not within_radius.empty:
#             st.write("Filtered Organizations within radius:", within_radius)
#         else:
#             st.write("No organizations found within the specified radius.")
#     else:
#         st.write("Zip code not found.")
















# ME 2 Apply the haversine function over the entire DataFrame
# if zip_code and radius:
#     try:
#         # Convert zip code to latitude and longitude
#         location = geolocator.geocode(zip_code, timeout=10, country_codes='US')
#         if location:
#             user_lat, user_lon = location.latitude, location.longitude
#             st.write("Location found:", (user_lat, user_lon))

#             # Filter data by category and remove rows with NaN in lat/lon
#             filtered_data = data[data['Category'].isin([selected_code])]
#             filtered_data = filtered_data.dropna(subset=['lat', 'lon'])

#             # Calculate distances in bulk using the haversine function
#             distances = haversine(user_lat, user_lon, filtered_data['lat'].values, filtered_data['lon'].values)
#             within_radius = filtered_data[distances <= radius]

#             # Display filtered data within radius
#             if not within_radius.empty:
#                 st.write("Filtered Organizations within radius:", within_radius)
#             else:
#                 st.write("No organizations found within the specified radius.")
#         else:
#             st.write("Zip code not found.")
#     except Exception as e:
#         st.write(f"Error with location services: {e}")







#MEEE
# if zip_code and radius:
#     try:
#         # Convert zip code to latitude and longitude
#         location = geolocator.geocode(zip_code, timeout=10, country_codes='US')
#         if location:
#             user_lat_lon = (location.latitude, location.longitude)
#             st.write("Location found:", user_lat_lon)

#             # Wrap `selected_code` in a list to make it compatible with `isin`
#             filtered_data = data[data['Category'].isin([selected_code])]
#             filtered_data = filtered_data.dropna(subset=['lat', 'lon'])
            
#             # Calculate distance and filter based on radius
#             within_radius = []
#             for _, row in filtered_data.iterrows():
#                 org_lat_lon = (row['lat'], row['lon'])
#                 distance = geodesic(user_lat_lon, org_lat_lon).miles
#                 if distance <= radius:
#                     within_radius.append(row)

#             # Display filtered data within radius
#             if within_radius:
#                 st.write("Filtered Organizations within radius:", pd.DataFrame(within_radius))
#             else:
#                 st.write("No organizations found within the specified radius.")
#         else:
#             st.write("Zip code not found.")
#     except Exception as e:
#         st.write(f"Error with location services: {e}")





# def haversine(lat1, lon1, lat2, lon2):
#     R = 3958.8  # Radius of Earth in miles
#     phi1 = np.radians(lat1)
#     phi2 = np.radians(lat2)
#     delta_phi = np.radians(lat2 - lat1)
#     delta_lambda = np.radians(lon2 - lon1)
#     a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     return R * c

# if zip_code and radius:
#     # Get latitude and longitude for the user's zip code
#     location = geolocator.query_postal_code(zip_code)
#     if not pd.isnull(location.latitude) and not pd.isnull(location.longitude):
#         user_lat, user_lon = location.latitude, location.longitude
#         st.write("User location found:", (user_lat, user_lon))

#         # Filter data based on radius
#         within_radius = []
#         for _, row in data.iterrows():
#             org_lat, org_lon = row['lat'], row['lon']
#             distance = haversine(user_lat, user_lon, org_lat, org_lon)
#             if distance <= radius:
#                 within_radius.append(row)

#         # Display filtered data within radius
#         if within_radius:
#             st.write("Filtered Organizations within radius:", pd.DataFrame(within_radius))
#         else:
#             st.write("No organizations found within the specified radius.")
#     else:
#         st.write("Invalid zip code.")

# ntee_categories = {
#     'Arts, Culture and Humanities' : {
#     'A01': 'Arts, Culture and Humanities: Alliance/Advocacy Organizations',
#     'A02': 'Arts, Culture and Humanities: Management & Technical Assistance',
#     'A03': 'Arts, Culture and Humanities: Professional Societies, Associations',
#     'A05': 'Arts, Culture and Humanities: Research Institutes and/or Public Policy Analysis',
#     'A11': 'Arts, Culture and Humanities: Single Organization Support',
#     'A12': 'Arts, Culture and Humanities: Fund Raising and/or Fund Distribution',
#     'A19': 'Arts, Culture and Humanities: Nonmonetary Support N.E.C.',
#     'A20': 'Arts, Culture and Humanities: Arts, Cultural Organizations - Multipurpose',
#     'A23': 'Arts, Culture and Humanities: Cultural, Ethnic Awareness',
#     'A24': 'Arts, Culture and Humanities: Folk Arts',
#     'A25': 'Arts, Culture and Humanities: Arts Education',
#     'A26': 'Arts, Culture and Humanities: Arts Council/Agency',
#     'A27': 'Arts, Culture and Humanities: Community Celebrations',
#     'A30': 'Arts, Culture and Humanities: Media, Communications Organizations',
#     'A31': 'Arts, Culture and Humanities: Film, Video',
#     'A32': 'Arts, Culture and Humanities: Television',
#     'A33': 'Arts, Culture and Humanities: Printing, Publishing',
#     'A34': 'Arts, Culture and Humanities: Radio',
#     'A40': 'Arts, Culture and Humanities: Visual Arts Organizations',
#     'A50': 'Arts, Culture and Humanities: Museum, Museum Activities',
#     'A51': 'Arts, Culture and Humanities: Art Museums',
#     'A52': "Arts, Culture and Humanities: Children's Museums",
#     'A53': 'Arts, Culture and Humanities: Folk Arts Museums',
#     'A54': 'Arts, Culture and Humanities: History Museums',
#     'A56': 'Arts, Culture and Humanities: Natural History, Natural Science Museums',
#     'A57': 'Arts, Culture and Humanities: Science and Technology Museums',
#     'A60': 'Arts, Culture and Humanities: Performing Arts Organizations',
#     'A61': 'Arts, Culture and Humanities: Performing Arts Centers',
#     'A62': 'Arts, Culture and Humanities: Dance',
#     'A63': 'Arts, Culture and Humanities: Ballet',
#     'A65': 'Arts, Culture and Humanities: Theater',
#     'A68': 'Arts, Culture and Humanities: Music',
#     'A69': 'Arts, Culture and Humanities: Symphony Orchestras',
#     'A6A': 'Arts, Culture and Humanities: Opera',
#     'A6B': 'Arts, Culture and Humanities: Singing, Choral',
#     'A6C': 'Arts, Culture and Humanities: Music Groups, Bands, Ensembles',
#     'A6E': 'Arts, Culture and Humanities: Performing Arts Schools',
#     'A70': 'Arts, Culture and Humanities: Humanities Organizations',
#     'A80': 'Arts, Culture and Humanities: Historical Societies, Related Historical Activities',
#     'A82': 'Arts, Culture and Humanities: Historical Societies & Historic Preservation',
#     'A84': 'Arts, Culture and Humanities: Commemorative Events',
#     'A90': 'Arts, Culture and Humanities: Arts Service Organizations and Activities',
#     'A99': 'Arts, Culture and Humanities: Arts, Culture, and Humanities N.E.C.'
#     },
#     'Education': {
#     'B01': 'Education: Alliance/Advocacy Organizations',
#     'B02': 'Education: Management & Technical Assistance',
#     'B03': 'Education: Professional Societies, Associations',
#     'B05': 'Education: Research Institutes and/or Public Policy Analysis',
#     'B11': 'Education: Single Organization Support',
#     'B12': 'Education: Fund Raising and/or Fund Distribution',
#     'B19': 'Education: Nonmonetary Support N.E.C.',
#     'B20': 'Education: Elementary, Secondary Education, K - 12',
#     'B21': 'Education: Kindergarten, Preschool, Nursery School, Early Admissions',
#     'B24': 'Education: Primary, Elementary Schools',
#     'B25': 'Education: Secondary, High School',
#     'B28': 'Education: Specialized Education Institutions',
#     'B30': 'Education: Vocational, Technical Schools',
#     'B40': 'Education: Higher Education Institutions',
#     'B41': 'Education: Community or Junior Colleges',
#     'B42': 'Education: Undergraduate College (4-year)',
#     'B43': 'Education: University or Technological Institute',
#     'B50': 'Education: Graduate, Professional Schools (Separate Entities)',
#     'B60': 'Education: Adult, Continuing Education',
#     'B70': 'Education: Libraries',
#     'B80': 'Education: Student Services, Organizations of Students',
#     'B82': 'Education: Scholarships, Student Financial Aid Services, Awards',
#     'B83': 'Education: Student Sororities, Fraternities',
#     'B84': 'Education: Alumni Associations',
#     'B90': 'Education: Educational Services and Schools - Other',
#     'B92': 'Education: Remedial Reading, Reading Encouragement',
#     'B94': 'Education: Parent/Teacher Group',
#     'B99': 'Education: Education N.E.C.'
#     },
#     'Environment':{
#     'C01': 'Environment: Alliance/Advocacy Organizations',
#     'C02': 'Environment: Management & Technical Assistance',
#     'C03': 'Environment: Professional Societies, Associations',
#     'C05': 'Environment: Research Institutes and/or Public Policy Analysis',
#     'C11': 'Environment: Single Organization Support',
#     'C12': 'Environment: Fund Raising and/or Fund Distribution',
#     'C19': 'Environment: Nonmonetary Support N.E.C.',
#     'C20': 'Environment: Pollution Abatement and Control Services',
#     'C27': 'Environment: Recycling Programs',
#     'C30': 'Environment: Natural Resources Conservation and Protection',
#     'C32': 'Environment: Water Resource, Wetlands Conservation and Management',
#     'C34': 'Environment: Land Resources Conservation',
#     'C35': 'Environment: Energy Resources Conservation and Development',
#     'C36': 'Environment: Forest Conservation',
#     'C40': 'Environment: Botanical, Horticultural, and Landscape Services',
#     'C41': 'Environment: Botanical Gardens, Arboreta and Botanical Organizations',
#     'C42': 'Environment: Garden Club, Horticultural Program',
#     'C50': 'Environment: Environmental Beautification and Aesthetics',
#     'C60': 'Environment: Environmental Education and Outdoor Survival Programs',
#     'C99': 'Environment: Environmental Quality, Protection, and Beautification N.E.C.'
#     },
#     'Animal-Related': {
#     'D01': 'Animal-Related: Alliance/Advocacy Organizations',
#     'D02': 'Animal-Related: Management & Technical Assistance',
#     'D03': 'Animal-Related: Professional Societies, Associations',
#     'D05': 'Animal-Related: Research Institutes and/or Public Policy Analysis',
#     'D11': 'Animal-Related: Single Organization Support',
#     'D12': 'Animal-Related: Fund Raising and/or Fund Distribution',
#     'D19': 'Animal-Related: Nonmonetary Support N.E.C.',
#     'D20': 'Animal-Related: Animal Protection and Welfare',
#     'D30': 'Animal-Related: Wildlife Preservation, Protection',
#     'D31': 'Animal-Related: Protection of Endangered Species',
#     'D32': 'Animal-Related: Bird Sanctuary, Preserve',
#     'D33': 'Animal-Related: Fisheries Resources',
#     'D34': 'Animal-Related: Wildlife Sanctuary, Refuge',
#     'D40': 'Animal-Related: Veterinary Services',
#     'D50': 'Animal-Related: Zoo, Zoological Society',
#     'D60': 'Animal-Related: Other Services - Specialty Animals',
#     'D61': 'Animal-Related: Animal Training, Behavior',
#     'D99': 'Animal-Related: Animal-Related N.E.C.'
#     },
#     'Health Care': {
#     'E01': 'Health Care: Alliance/Advocacy Organizations',
#     'E02': 'Health Care: Management & Technical Assistance',
#     'E03': 'Health Care: Professional Societies, Associations',
#     'E05': 'Health Care: Research Institutes and/or Public Policy Analysis',
#     'E11': 'Health Care: Single Organization Support',
#     'E12': 'Health Care: Fund Raising and/or Fund Distribution',
#     'E19': 'Health Care: Nonmonetary Support N.E.C.',
#     'E20': 'Health Care: Hospitals and Related Primary Medical Care Facilities',
#     'E21': 'Health Care: Community Health Systems',
#     'E22': 'Health Care: Hospital, General',
#     'E24': 'Health Care: Hospital, Specialty',
#     'E30': 'Health Care: Health Treatment Facilities, Primarily Outpatient',
#     'E31': 'Health Care: Group Health Practice (Health Maintenance Organizations)',
#     'E32': 'Health Care: Ambulatory Health Center, Community Clinic',
#     'E40': 'Health Care: Reproductive Health Care Facilities and Allied Services',
#     'E42': 'Health Care: Family Planning Centers',
#     'E50': 'Health Care: Rehabilitative Medical Services',
#     'E60': 'Health Care: Health Support Services',
#     'E61': 'Health Care: Blood Supply Related',
#     'E62': 'Health Care: Ambulance, Emergency Medical Transport Services',
#     'E65': 'Health Care: Organ and Tissue Banks',
#     'E70': 'Health Care: Public Health Program (Includes General Health and Wellness Promotion Services)',
#     'E80': 'Health Care: Health, General and Financing',
#     'E86': 'Health Care: Patient Services - Entertainment, Recreation',
#     'E90': 'Health Care: Nursing Services (General)',
#     'E91': 'Health Care: Nursing, Convalescent Facilities',
#     'E92': 'Health Care: Home Health Care',
#     'E99': 'Health Care: Health - General and Rehabilitative N.E.C.'
#     },
#     'Mental Health & Crisis Intervention': {
#     'F01': 'Mental Health & Crisis Intervention: Alliance/Advocacy Organizations',
#     'F02': 'Mental Health & Crisis Intervention: Management & Technical Assistance',
#     'F03': 'Mental Health & Crisis Intervention: Professional Societies, Associations',
#     'F05': 'Mental Health & Crisis Intervention: Research Institutes and/or Public Policy Analysis',
#     'F11': 'Mental Health & Crisis Intervention: Single Organization Support',
#     'F12': 'Mental Health & Crisis Intervention: Fund Raising and/or Fund Distribution',
#     'F19': 'Mental Health & Crisis Intervention: Nonmonetary Support N.E.C.',
#     'F20': 'Mental Health & Crisis Intervention: Alcohol, Drug and Substance Abuse, Dependency Prevention and Treatment',
#     'F21': 'Mental Health & Crisis Intervention: Alcohol, Drug Abuse, Prevention Only',
#     'F22': 'Mental Health & Crisis Intervention: Alcohol, Drug Abuse, Treatment Only',
#     'F30': 'Mental Health & Crisis Intervention: Mental Health Treatment',
#     'F31': 'Mental Health & Crisis Intervention: Psychiatric, Mental Health Hospital',
#     'F32': 'Mental Health & Crisis Intervention: Community Mental Health Center',
#     'F33': 'Mental Health & Crisis Intervention: Residential Mental Health Treatment',
#     'F40': 'Mental Health & Crisis Intervention: Hot Line, Crisis Intervention Services',
#     'F42': 'Mental Health & Crisis Intervention: Rape Victim Services',
#     'F50': 'Mental Health & Crisis Intervention: Addictive Disorders N.E.C.',
#     'F52': 'Mental Health & Crisis Intervention: Smoking Addiction',
#     'F53': 'Mental Health & Crisis Intervention: Eating Disorder, Addiction',
#     'F54': 'Mental Health & Crisis Intervention: Gambling Addiction',
#     'F60': 'Mental Health & Crisis Intervention: Counseling, Support Groups',
#     'F70': 'Mental Health & Crisis Intervention: Mental Health Disorders',
#     'F80': 'Mental Health & Crisis Intervention: Mental Health Association',
#     'F99': 'Mental Health & Crisis Intervention: Mental Health, Crisis Intervention N.E.C.'
#     },
#     'Voluntary Health Associations & Medical Disciplines':{
#     'G01': 'Voluntary Health Associations & Medical Disciplines: Alliance/Advocacy Organizations',
#     'G02': 'Voluntary Health Associations & Medical Disciplines: Management & Technical Assistance',
#     'G03': 'Voluntary Health Associations & Medical Disciplines: Professional Societies, Associations',
#     'G05': 'Voluntary Health Associations & Medical Disciplines: Research Institutes and/or Public Policy Analysis',
#     'G11': 'Voluntary Health Associations & Medical Disciplines: Single Organization Support',
#     'G12': 'Voluntary Health Associations & Medical Disciplines: Fund Raising and/or Fund Distribution',
#     'G19': 'Voluntary Health Associations & Medical Disciplines: Support N.E.C.',
#     'G20': 'Voluntary Health Associations & Medical Disciplines: Birth Defects and Genetic Diseases',
#     'G25': 'Voluntary Health Associations & Medical Disciplines: Down Syndrome',
#     'G30': 'Voluntary Health Associations & Medical Disciplines: Cancer',
#     'G32': 'Voluntary Health Associations & Medical Disciplines: Breast Cancer',
#     'G40': 'Voluntary Health Associations & Medical Disciplines: Diseases of Specific Organs',
#     'G41': 'Voluntary Health Associations & Medical Disciplines: Eye Diseases, Blindness and Vision Impairments',
#     'G42': 'Voluntary Health Associations & Medical Disciplines: Ear and Throat Diseases',
#     'G43': 'Voluntary Health Associations & Medical Disciplines: Heart and Circulatory System Diseases, Disorders',
#     'G44': 'Voluntary Health Associations & Medical Disciplines: Kidney Disease',
#     'G45': 'Voluntary Health Associations & Medical Disciplines: Lung Disease',
#     'G48': 'Voluntary Health Associations & Medical Disciplines: Brain Disorders',
#     'G50': 'Voluntary Health Associations & Medical Disciplines: Nerve, Muscle and Bone Diseases',
#     'G51': 'Voluntary Health Associations & Medical Disciplines: Arthritis',
#     'G54': 'Voluntary Health Associations & Medical Disciplines: Epilepsy',
#     'G60': 'Voluntary Health Associations & Medical Disciplines: Allergy Related Diseases',
#     'G61': 'Voluntary Health Associations & Medical Disciplines: Asthma',
#     'G70': 'Voluntary Health Associations & Medical Disciplines: Digestive Diseases, Disorders',
#     'G80': 'Voluntary Health Associations & Medical Disciplines: Specifically Named Diseases',
#     'G81': 'Voluntary Health Associations & Medical Disciplines: AIDS',
#     'G83': "Voluntary Health Associations & Medical Disciplines: Alzheimer's Disease",
#     'G84': 'Voluntary Health Associations & Medical Disciplines: Autism',
#     'G90': 'Voluntary Health Associations & Medical Disciplines: Medical Disciplines',
#     'G92': 'Voluntary Health Associations & Medical Disciplines: Biomedicine, Bioengineering',
#     'G94': 'Voluntary Health Associations & Medical Disciplines: Geriatrics',
#     'G96': 'Voluntary Health Associations & Medical Disciplines: Neurology, Neuroscience',
#     'G98': 'Voluntary Health Associations & Medical Disciplines: Pediatrics',
#     'G99': 'Voluntary Health Associations & Medical Disciplines: Voluntary Health Associations & Medical Disciplines N.E.C.',
#     'G9B': 'Voluntary Health Associations & Medical Disciplines: Surgery Specialties'
#     },
#     'Medical Research': {
#     'H01': 'Medical Research: Alliance/Advocacy Organizations',
#     'H02': 'Medical Research: Management & Technical Assistance',
#     'H03': 'Medical Research: Professional Societies, Associations',
#     'H05': 'Medical Research: Research Institutes and/or Public Policy Analysis',
#     'H11': 'Medical Research: Single Organization Support',
#     'H12': 'Medical Research: Fund Raising and/or Fund Distribution',
#     'H19': 'Medical Research: Support N.E.C.',
#     'H20': 'Medical Research: Birth Defects, Genetic Diseases Research',
#     'H25': 'Medical Research: Down Syndrome Research',
#     'H30': 'Medical Research: Cancer Research',
#     'H32': 'Medical Research: Breast Cancer Research',
#     'H40': 'Medical Research: Diseases of Specific Organ Research',
#     'H41': 'Medical Research: Eye Diseases, Blindness & Vision Impairments Research',
#     'H42': 'Medical Research: Ear and Throat Research',
#     'H43': 'Medical Research: Heart, Circulatory Research',
#     'H44': 'Medical Research: Kidney Diseases Research',
#     'H45': 'Medical Research: Lung Diseases Research',
#     'H48': 'Medical Research: Brain Disorders Research',
#     'H50': 'Medical Research: Nerve, Muscle, Bone Diseases Research',
#     'H51': 'Medical Research: Arthritis Research',
#     'H54': 'Medical Research: Epilepsy Research',
#     'H60': 'Medical Research: Allergy Related Disease Research',
#     'H61': 'Medical Research: Asthma Research',
#     'H70': 'Medical Research: Digestive Disease, Disorder Research',
#     'H80': 'Medical Research: Specifically Named Diseases Research',
#     'H81': 'Medical Research: AIDS Research',
#     'H83': "Medical Research: Alzheimer's Disease Research",
#     'H84': 'Medical Research: Autism Research',
#     'H90': 'Medical Research: Medical Disciplines Research',
#     'H92': 'Medical Research: Biomedicine, Bioengineering Research',
#     'H94': 'Medical Research: Geriatrics Research',
#     'H96': 'Medical Research: Neurology, Neuroscience Research',
#     'H98': 'Medical Research: Pediatrics Research',
#     'H99': 'Medical Research: Medical Research N.E.C.',
#     'H9B': 'Medical Research: Surgical Specialties Research'
#     },
#     'Crime & Legal-Related': {
#     'I01': 'Crime & Legal-Related: Alliance/Advocacy',
#     'I02': 'Crime & Legal-Related: Management & Technical Assistance',
#     'I03': 'Crime & Legal-Related: Professional Societies, Associations',
#     'I05': 'Crime & Legal-Related: Research Institutes and Public Policy Analysis',
#     'I11': 'Crime & Legal-Related: Single Organization Support',
#     'I12': 'Crime & Legal-Related: Fund Raising and Fund Distribution',
#     'I19': 'Crime & Legal-Related: Support N.E.C.',
#     'I20': 'Crime & Legal-Related: Crime Prevention',
#     'I21': 'Crime & Legal-Related: Delinquency Prevention',
#     'I23': 'Crime & Legal-Related: Drunk Driving Related',
#     'I30': 'Crime & Legal-Related: Correctional Facilities N.E.C.',
#     'I31': 'Crime & Legal-Related: Half-Way House for Offenders, Ex-Offenders',
#     'I40': 'Crime & Legal-Related: Rehabilitation Services for Offenders',
#     'I43': 'Crime & Legal-Related: Inmate Support',
#     'I44': 'Crime & Legal-Related: Prison Alternatives',
#     'I50': 'Crime & Legal-Related: Administration of Justice, Courts',
#     'I51': 'Crime & Legal-Related: Dispute Resolution, Mediation Services',
#     'I60': 'Crime & Legal-Related: Law Enforcement',
#     'I70': 'Crime & Legal-Related: Protection Against Abuse',
#     'I71': 'Crime & Legal-Related: Spouse Abuse, Prevention',
#     'I72': 'Crime & Legal-Related: Child Abuse, Prevention',
#     'I73': 'Crime & Legal-Related: Sexual Abuse, Prevention',
#     'I80': 'Crime & Legal-Related: Legal Services',
#     'I83': 'Crime & Legal-Related: Public Interest Law',
#     'I99': 'Crime & Legal-Related: Crime, Legal Related N.E.C.'
#     },
#     'Employment': {
#     'J01': 'Employment: Alliance/Advocacy Organizations',
#     'J02': 'Employment: Management & Technical Assistance',
#     'J03': 'Employment: Professional Societies, Associations',
#     'J05': 'Employment: Research Institutes and/or Public Policy Analysis',
#     'J11': 'Employment: Single Organization Support',
#     'J12': 'Employment: Fund Raising and Fund Distribution',
#     'J19': 'Employment: Support N.E.C.',
#     'J20': 'Employment: Employment Procurement Assistance',
#     'J21': 'Employment: Vocational Counseling',
#     'J22': 'Employment: Job Training',
#     'J30': 'Employment: Vocational Rehabilitation',
#     'J32': 'Employment: Goodwill Industries',
#     'J33': 'Employment: Sheltered Employment',
#     'J40': 'Employment: Labor Unions',
#     'J99': 'Employment: Employment N.E.C.'
#     },
#     'Food, Agriculture and Nutrition': {
#     'K01': 'Food, Agriculture and Nutrition: Alliance/Advocacy',
#     'K02': 'Food, Agriculture and Nutrition: Management & Technical Assistance',
#     'K03': 'Food, Agriculture and Nutrition: Professional Societies & Associations',
#     'K05': 'Food, Agriculture and Nutrition: Research Institutes and Public Policy Analysis',
#     'K11': 'Food, Agriculture and Nutrition: Single Organization Support',
#     'K12': 'Food, Agriculture and Nutrition: Fund Raising and Fund Distribution',
#     'K19': 'Food, Agriculture and Nutrition: Support N.E.C.',
#     'K20': 'Food, Agriculture and Nutrition: Agricultural Programs',
#     'K25': 'Food, Agriculture and Nutrition: Farmland Preservation',
#     'K26': 'Food, Agriculture and Nutrition: Animal Husbandry',
#     'K28': 'Food, Agriculture and Nutrition: Farm Bureau, Grange',
#     'K30': 'Food, Agriculture and Nutrition: Food Programs',
#     'K31': 'Food, Agriculture and Nutrition: Food Banks, Pantries',
#     'K34': 'Food, Agriculture and Nutrition: Congregate Meals',
#     'K35': 'Food, Agriculture and Nutrition: Soup Kitchens',
#     'K36': 'Food, Agriculture and Nutrition: Meals on Wheels',
#     'K40': 'Food, Agriculture and Nutrition: Nutrition',
#     'K50': 'Food, Agriculture and Nutrition: Home Economics',
#     'K6A': 'Food, Agriculture and Nutrition: Confectionary and Nut Stores',
#     'K6C': 'Food, Agriculture and Nutrition: Caterers',
#     'K6D': 'Food, Agriculture and Nutrition: Mobile Food Services',
#     'K6E': 'Food, Agriculture and Nutrition: Drinking Places (Alcoholic Beverages)',
#     'K6F': 'Food, Agriculture and Nutrition: Snack and Nonalcoholic Beverage Bars',
#     'K90': 'Food, Agriculture and Nutrition: Limited-Service Restaurants',
#     'K91': 'Food, Agriculture and Nutrition: Supermarkets and Other Grocery (except Convenience) Stores',
#     'K92': 'Food, Agriculture and Nutrition: Convenience Stores',
#     'K93': 'Food, Agriculture and Nutrition: Fruit and Vegetable Markets',
#     'K94': 'Food, Agriculture and Nutrition: All Other Specialty Food Stores',
#     'K95': 'Food, Agriculture and Nutrition: Food (Health) Supplement Stores',
#     'K96': 'Food, Agriculture and Nutrition: Warehouse Clubs and Supercenters',
#     'K97': 'Food, Agriculture and Nutrition: Food Service Contractors',
#     'K98': 'Food, Agriculture and Nutrition: Full-Service Restaurants',
#     'K99': 'Food, Agriculture and Nutrition: Food, Agriculture, and Nutrition N.E.C.'
#     },
#     'Housing & Shelter': {
#     'L01': 'Housing & Shelter: Alliance/Advocacy Organizations',
#     'L02': 'Housing & Shelter: Management & Technical Assistance',
#     'L03': 'Housing & Shelter: Professional Societies, Associations',
#     'L05': 'Housing & Shelter: Research Institutes and Public Policy Analysis',
#     'L11': 'Housing & Shelter: Single Organization Support',
#     'L12': 'Housing & Shelter: Fund Raising and Fund Distribution',
#     'L19': 'Housing & Shelter: Support N.E.C.',
#     'L20': 'Housing & Shelter: Housing Development, Construction, Management',
#     'L21': 'Housing & Shelter: Low-Income & Subsidized Rental Housing',
#     'L22': 'Housing & Shelter: Senior Citizens\' Housing/Retirement Communities',
#     'L24': 'Housing & Shelter: Independent Housing for People with Disabilities',
#     'L25': 'Housing & Shelter: Housing Rehabilitation',
#     'L30': 'Housing & Shelter: Housing Search Assistance',
#     'L40': 'Housing & Shelter: Temporary Housing',
#     'L41': 'Housing & Shelter: Homeless Shelters',
#     'L50': 'Housing & Shelter: Homeowners & Tenants Associations',
#     'L80': 'Housing & Shelter: Housing Support',
#     'L81': 'Housing & Shelter: Home Improvement and Repairs',
#     'L82': 'Housing & Shelter: Housing Expense Reduction Support',
#     'L99': 'Housing & Shelter: Housing, Shelter N.E.C.'
#     },
#     'Public Safety, Disaster Preparedness & Relief': {
#     'M01': 'Public Safety, Disaster Preparedness & Relief: Alliance/Advocacy',
#     'M02': 'Public Safety, Disaster Preparedness & Relief: Management & Technical Assistance',
#     'M03': 'Public Safety, Disaster Preparedness & Relief: Professional Societies, Associations',
#     'M05': 'Public Safety, Disaster Preparedness & Relief: Research Institutes and Public Policy Analysis',
#     'M11': 'Public Safety, Disaster Preparedness & Relief: Single Organization Support',
#     'M12': 'Public Safety, Disaster Preparedness & Relief: Fund Raising and Fund Distribution',
#     'M19': 'Public Safety, Disaster Preparedness & Relief: Support N.E.C.',
#     'M20': 'Public Safety, Disaster Preparedness & Relief: Disaster Preparedness and Relief Services',
#     'M23': 'Public Safety, Disaster Preparedness & Relief: Search and Rescue Squads',
#     'M24': 'Public Safety, Disaster Preparedness & Relief: Fire Prevention',
#     'M40': 'Public Safety, Disaster Preparedness & Relief: Safety Education',
#     'M41': 'Public Safety, Disaster Preparedness & Relief: First Aid',
#     'M42': 'Public Safety, Disaster Preparedness & Relief: Automotive Safety',
#     'M60': 'Public Safety, Disaster Preparedness & Relief: Public Safety Benevolent Associations',
#     'M99': 'Public Safety, Disaster Preparedness & Relief: Public Safety, Disaster Preparedness & Relief N.E.C.'
#     },
#     'Recreation & Sports': {
#     'N01': 'Recreation & Sports: Alliance/Advocacy',
#     'N02': 'Recreation & Sports: Management & Technical Assistance',
#     'N03': 'Recreation & Sports: Professional Societies, Associations',
#     'N05': 'Recreation & Sports: Research Institutes and/or Public Policy Analysis',
#     'N11': 'Recreation & Sports: Single Organization Support',
#     'N12': 'Recreation & Sports: Fund Raising and Fund Distribution',
#     'N19': 'Recreation & Sports: Support N.E.C.',
#     'N20': 'Recreation & Sports: Camps',
#     'N2A': 'Recreation & Sports: RV (Recreation Vehicle) Parks and Campgrounds',
#     'N2B': 'Recreation & Sports: Recreational and Vacation Camps (Except Campgrounds)',
#     'N30': 'Recreation & Sports: Physical Fitness and Community Recreational Facilities',
#     'N31': 'Recreation & Sports: Community Recreational Centers',
#     'N32': 'Recreation & Sports: Parks and Playgrounds',
#     'N40': 'Recreation & Sports: Sports Associations & Training Facilities',
#     'N50': 'Recreation & Sports: Recreational Clubs',
#     'N52': 'Recreation & Sports: Fairs',
#     'N60': 'Recreation & Sports: Amateur Sports',
#     'N61': 'Recreation & Sports: Fishing, Hunting',
#     'N62': 'Recreation & Sports: Basketball',
#     'N63': 'Recreation & Sports: Baseball, Softball',
#     'N64': 'Recreation & Sports: Soccer',
#     'N65': 'Recreation & Sports: Football',
#     'N66': 'Recreation & Sports: Racquet Sports',
#     'N67': 'Recreation & Sports: Swimming & Other Water Recreation',
#     'N68': 'Recreation & Sports: Winter Sports',
#     'N69': 'Recreation & Sports: Equestrian',
#     'N6A': 'Recreation & Sports: Golf',
#     'N70': 'Recreation & Sports: Amateur Sports Competitions',
#     'N71': 'Recreation & Sports: Olympics',
#     'N72': 'Recreation & Sports: Special Olympics',
#     'N80': 'Recreation & Sports: Professional Athletic Leagues',
#     'N99': 'Recreation & Sports: Recreation & Sports N.E.C.'
#     },
#     'Youth Development': {
#     'O01': 'Youth Development: Alliance/Advocacy',
#     'O02': 'Youth Development: Management & Technical Assistance',
#     'O03': 'Youth Development: Professional Societies, Associations',
#     'O05': 'Youth Development: Research Institutes and Public Policy Analysis',
#     'O11': 'Youth Development: Single Organization Support',
#     'O12': 'Youth Development: Fund Raising and Fund Distribution',
#     'O19': 'Youth Development: Support N.E.C.',
#     'O20': 'Youth Development: Youth Centers, Clubs',
#     'O21': 'Youth Development: Boys Clubs',
#     'O22': 'Youth Development: Girls Clubs',
#     'O23': 'Youth Development: Boys and Girls Clubs',
#     'O30': 'Youth Development: Adult, Child Matching Programs',
#     'O31': 'Youth Development: Big Brothers, Big Sisters',
#     'O40': 'Youth Development: Scouting',
#     'O41': 'Youth Development: Boy Scouts of America',
#     'O42': 'Youth Development: Girl Scouts of the U.S.A.',
#     'O43': 'Youth Development: Camp Fire',
#     'O50': 'Youth Development: Youth Development Programs',
#     'O51': 'Youth Development: Youth Community Service Clubs',
#     'O52': 'Youth Development: Agricultural',
#     'O53': 'Youth Development: Business',
#     'O54': 'Youth Development: Citizenship',
#     'O55': 'Youth Development: Religious Leadership',
#     'O99': 'Youth Development: Youth Development N.E.C.'
#     },
#     'Human Services': {
#     'P01': 'Human Services: Alliance/Advocacy',
#     'P02': 'Human Services: Management & Technical Assistance',
#     'P03': 'Human Services: Professional Societies, Associations',
#     'P05': 'Human Services: Research Institutes and Public Policy Analysis',
#     'P11': 'Human Services: Single Organization Support',
#     'P12': 'Human Services: Fund Raising and Fund Distribution',
#     'P19': 'Human Services: Support N.E.C.',
#     'P20': 'Human Services: Human Service Organizations',
#     'P21': 'Human Services: American Red Cross',
#     'P22': 'Human Services: Urban League',
#     'P24': 'Human Services: Salvation Army',
#     'P26': 'Human Services: Volunteers of America',
#     'P27': 'Human Services: Young Men\'s or Women\'s Associations',
#     'P28': 'Human Services: Neighborhood Centers',
#     'P29': 'Human Services: Thrift Shops',
#     'P30': 'Human Services: Children\'s, Youth Services',
#     'P31': 'Human Services: Adoption',
#     'P32': 'Human Services: Foster Care',
#     'P33': 'Human Services: Child Day Care',
#     'P40': 'Human Services: Family Services',
#     'P42': 'Human Services: Single Parent Agencies',
#     'P43': 'Human Services: Family Violence Shelters',
#     'P44': 'Human Services: In-Home Assistance',
#     'P45': 'Human Services: Family Services for Adolescent Parents',
#     'P46': 'Human Services: Family Counseling',
#     'P47': 'Human Services: Pregnancy Centers',
#     'P50': 'Human Services: Personal Social Services',
#     'P51': 'Human Services: Financial Counseling',
#     'P52': 'Human Services: Transportation Assistance',
#     'P58': 'Human Services: Gift Distribution',
#     'P60': 'Human Services: Emergency Assistance',
#     'P61': 'Human Services: Travelers\' Aid',
#     'P62': 'Human Services: Victims\' Services',
#     'P70': 'Human Services: Residential Care & Adult Day Programs',
#     'P71': 'Human Services: Adult Day Care',
#     'P73': 'Human Services: Group Homes',
#     'P74': 'Human Services: Hospices',
#     'P75': 'Human Services: Supportive Housing for Older Adults',
#     'P80': 'Human Services: Centers to Support the Independence of Specific Populations',
#     'P81': 'Human Services: Senior Centers',
#     'P82': 'Human Services: Developmentally Disabled Centers',
#     'P84': 'Human Services: Ethnic, Immigrant Centers',
#     'P85': 'Human Services: Homeless Centers',
#     'P86': 'Human Services: Blind/Visually Impaired Centers',
#     'P87': 'Human Services: Deaf/Hearing Impaired Centers',
#     'P88': 'Human Services: LGBT Centers',
#     'P99': 'Human Services: Human Services N.E.C.'
#     },
#     'International, Foreign Affairs and National Security': {
#     'Q01': 'International, Foreign Affairs and National Security: Alliance/Advocacy Organizations',
#     'Q02': 'International, Foreign Affairs and National Security: Management & Technical Assistance',
#     'Q03': 'International, Foreign Affairs and National Security: Professional Societies, Associations',
#     'Q05': 'International, Foreign Affairs and National Security: Research Institutes and Public Policy Analysis',
#     'Q11': 'International, Foreign Affairs and National Security: Single Organization Support',
#     'Q12': 'International, Foreign Affairs and National Security: Fund Raising and Fund Distribution',
#     'Q19': 'International, Foreign Affairs and National Security: Support N.E.C.',
#     'Q20': 'International, Foreign Affairs and National Security: Promotion of International Understanding',
#     'Q21': 'International, Foreign Affairs and National Security: International Cultural Exchange',
#     'Q22': 'International, Foreign Affairs and National Security: International Academic Exchange',
#     'Q23': 'International, Foreign Affairs and National Security: International Exchange, N.E.C.',
#     'Q30': 'International, Foreign Affairs and National Security: International Development',
#     'Q31': 'International, Foreign Affairs and National Security: International Agricultural Development',
#     'Q32': 'International, Foreign Affairs and National Security: International Economic Development',
#     'Q33': 'International, Foreign Affairs and National Security: International Relief',
#     'Q34': 'International, Foreign Affairs and National Security: International Educational Development',
#     'Q35': 'International, Foreign Affairs and National Security: International Democracy & Civil Society Development',
#     'Q36': 'International, Foreign Affairs and National Security: International Science & Technology Development',
#     'Q38': 'International, Foreign Affairs and National Security: International Environment, Population & Sustainability',
#     'Q39': 'International, Foreign Affairs and National Security: International Health Development',
#     'Q40': 'International, Foreign Affairs and National Security: International Peace and Security',
#     'Q41': 'International, Foreign Affairs and National Security: Arms Control, Peace',
#     'Q42': 'International, Foreign Affairs and National Security: United Nations Association',
#     'Q43': 'International, Foreign Affairs and National Security: National Security',
#     'Q50': 'International, Foreign Affairs and National Security: International Affairs, Foreign Policy, & Globalization',
#     'Q51': 'International, Foreign Affairs and National Security: International Economic & Trade Policy',
#     'Q70': 'International, Foreign Affairs and National Security: International Human Rights',
#     'Q71': 'International, Foreign Affairs and National Security: International Migration, Refugee Issues',
#     'Q99': 'International, Foreign Affairs and National Security: International, Foreign Affairs, and National Security N.E.C.'
#     },
#     'Civil Rights, Social Action, Advocacy': {
#     'R01': 'Civil Rights, Social Action, Advocacy: Alliance/Advocacy',
#     'R02': 'Civil Rights, Social Action, Advocacy: Management & Technical Assistance',
#     'R03': 'Civil Rights, Social Action, Advocacy: Professional Societies, Associations',
#     'R05': 'Civil Rights, Social Action, Advocacy: Research Institutes and Public Policy Analysis',
#     'R11': 'Civil Rights, Social Action, Advocacy: Single Organization Support',
#     'R12': 'Civil Rights, Social Action, Advocacy: Fund Raising and Fund Distribution',
#     'R19': 'Civil Rights, Social Action, Advocacy: Support N.E.C.',
#     'R20': 'Civil Rights, Social Action, Advocacy: Civil Rights',
#     'R21': 'Civil Rights, Social Action, Advocacy: Immigrant’s Rights',
#     'R22': 'Civil Rights, Social Action, Advocacy: Minority Rights',
#     'R23': 'Civil Rights, Social Action, Advocacy: Disabled Persons\' Rights',
#     'R24': 'Civil Rights, Social Action, Advocacy: Women\'s Rights',
#     'R25': 'Civil Rights, Social Action, Advocacy: Seniors\' Rights',
#     'R26': 'Civil Rights, Social Action, Advocacy: Lesbian, Gay Rights',
#     'R27': 'Civil Rights, Social Action, Advocacy: Patient’s Rights',
#     'R28': 'Civil Rights, Social Action, Advocacy: Children’s Rights',
#     'R29': 'Civil Rights, Social Action, Advocacy: Employee & Workers Rights',
#     'R30': 'Civil Rights, Social Action, Advocacy: Intergroup, Race Relations',
#     'R40': 'Civil Rights, Social Action, Advocacy: Voter Education, Registration',
#     'R60': 'Civil Rights, Social Action, Advocacy: Civil Liberties',
#     'R61': 'Civil Rights, Social Action, Advocacy: Reproductive Rights',
#     'R62': 'Civil Rights, Social Action, Advocacy: Right to Life',
#     'R63': 'Civil Rights, Social Action, Advocacy: Censorship, Freedom of Speech and Press',
#     'R65': 'Civil Rights, Social Action, Advocacy: Freedom of Religion Issues',
#     'R67': 'Civil Rights, Social Action, Advocacy: Right to Die, Euthanasia',
#     'R99': 'Civil Rights, Social Action, Advocacy: Civil Rights, Social Action, Advocacy N.E.C.'
#     },
#     'Community Improvement, Capacity Building': {
#     'S01': 'Community Improvement, Capacity Building: Alliance/Advocacy',
#     'S02': 'Community Improvement, Capacity Building: Management & Technical Assistance',
#     'S03': 'Community Improvement, Capacity Building: Professional Societies, Associations',
#     'S05': 'Community Improvement, Capacity Building: Research Institutes and Public Policy Analysis',
#     'S11': 'Community Improvement, Capacity Building: Single Organization Support',
#     'S12': 'Community Improvement, Capacity Building: Fund Raising and Fund Distribution',
#     'S19': 'Community Improvement, Capacity Building: Support N.E.C.',
#     'S20': 'Community Improvement, Capacity Building: Community, Neighborhood Development',
#     'S21': 'Community Improvement, Capacity Building: Community Coalitions',
#     'S22': 'Community Improvement, Capacity Building: Neighborhood, Block Associations',
#     'S30': 'Community Improvement, Capacity Building: Economic Development',
#     'S31': 'Community Improvement, Capacity Building: Urban, Community Economic Development',
#     'S32': 'Community Improvement, Capacity Building: Rural Economic Development',
#     'S40': 'Community Improvement, Capacity Building: Business and Industry',
#     'S41': 'Community Improvement, Capacity Building: Chambers of Commerce & Business Leagues',
#     'S43': 'Community Improvement, Capacity Building: Small Business Development',
#     'S46': 'Community Improvement, Capacity Building: Boards of Trade',
#     'S47': 'Community Improvement, Capacity Building: Real Estate Associations',
#     'S50': 'Community Improvement, Capacity Building: Nonprofit Management',
#     'S80': 'Community Improvement, Capacity Building: Community Service Clubs',
#     'S81': 'Community Improvement, Capacity Building: Women\'s Service Clubs',
#     'S82': 'Community Improvement, Capacity Building: Men\'s Service Clubs',
#     'S99': 'Community Improvement, Capacity Building: Community Improvement, Capacity Building N.E.C.'
#     },
#     'Philanthropy, Voluntarism and Grantmaking Foundations': {
#     'T01': 'Philanthropy, Voluntarism and Grantmaking Foundations: Alliance/Advocacy',
#     'T02': 'Philanthropy, Voluntarism and Grantmaking Foundations: Management & Technical Assistance',
#     'T03': 'Philanthropy, Voluntarism and Grantmaking Foundations: Professional Societies, Associations',
#     'T05': 'Philanthropy, Voluntarism and Grantmaking Foundations: Research Institutes and Public Policy Analysis',
#     'T11': 'Philanthropy, Voluntarism and Grantmaking Foundations: Single Organization Support',
#     'T12': 'Philanthropy, Voluntarism and Grantmaking Foundations: Fund Raising and Fund Distribution',
#     'T19': 'Philanthropy, Voluntarism and Grantmaking Foundations: Support N.E.C.',
#     'T20': 'Philanthropy, Voluntarism and Grantmaking Foundations: Private Grantmaking Foundations',
#     'T21': 'Philanthropy, Voluntarism and Grantmaking Foundations: Corporate Foundations',
#     'T22': 'Philanthropy, Voluntarism and Grantmaking Foundations: Private Independent Foundations',
#     'T23': 'Philanthropy, Voluntarism and Grantmaking Foundations: Private Operating Foundations',
#     'T30': 'Philanthropy, Voluntarism and Grantmaking Foundations: Public Foundations',
#     'T31': 'Philanthropy, Voluntarism and Grantmaking Foundations: Community Foundations',
#     'T40': 'Philanthropy, Voluntarism and Grantmaking Foundations: Voluntarism Promotion',
#     'T50': 'Philanthropy, Voluntarism and Grantmaking Foundations: Philanthropy, Charity, Voluntarism Promotion',
#     'T70': 'Philanthropy, Voluntarism and Grantmaking Foundations: Federated Giving Programs',
#     'T90': 'Philanthropy, Voluntarism and Grantmaking Foundations: Named Trusts N.E.C.',
#     'T99': 'Philanthropy, Voluntarism and Grantmaking Foundations: Philanthropy, Voluntarism, and Grantmaking Foundations N.E.C.'
#     },
#     'Science and Technology': {
#     'U01': 'Science and Technology: Alliance/Advocacy',
#     'U02': 'Science and Technology: Management & Technical Assistance',
#     'U03': 'Science and Technology: Professional Societies, Associations',
#     'U05': 'Science and Technology: Research Institutes and Public Policy Analysis',
#     'U11': 'Science and Technology: Single Organization Support',
#     'U12': 'Science and Technology: Fund Raising and Fund Distribution',
#     'U19': 'Science and Technology: Support N.E.C.',
#     'U20': 'Science and Technology: General Science',
#     'U21': 'Science and Technology: Marine Science and Oceanography',
#     'U30': 'Science and Technology: Physical & Earth Sciences',
#     'U31': 'Science and Technology: Astronomy',
#     'U33': 'Science and Technology: Chemistry, Chemical Engineering',
#     'U34': 'Science and Technology: Mathematics',
#     'U36': 'Science and Technology: Geology',
#     'U40': 'Science and Technology: Engineering and Technology',
#     'U41': 'Science and Technology: Computer Science',
#     'U42': 'Science and Technology: Engineering',
#     'U50': 'Science and Technology: Biological & Life Sciences',
#     'U99': 'Science and Technology: Science and Technology N.E.C.'
#     },
#     'Social Science': {
#     'V01': 'Social Science: Alliance/Advocacy',
#     'V02': 'Social Science: Management & Technical Assistance',
#     'V03': 'Social Science: Professional Societies, Associations',
#     'V05': 'Social Science: Research Institutes and Public Policy Analysis',
#     'V11': 'Social Science: Single Organization Support',
#     'V12': 'Social Science: Fund Raising and Fund Distribution',
#     'V19': 'Social Science: Support N.E.C.',
#     'V20': 'Social Science: Social Science',
#     'V21': 'Social Science: Anthropology, Sociology',
#     'V22': 'Social Science: Economics',
#     'V23': 'Social Science: Behavioral Science',
#     'V24': 'Social Science: Political Science',
#     'V25': 'Social Science: Population Studies',
#     'V26': 'Social Science: Law, Jurisprudence',
#     'V30': 'Social Science: Interdisciplinary Research',
#     'V31': 'Social Science: Black Studies',
#     'V32': 'Social Science: Women\'s Studies',
#     'V33': 'Social Science: Ethnic Studies',
#     'V34': 'Social Science: Urban Studies',
#     'V35': 'Social Science: International Studies',
#     'V36': 'Social Science: Gerontology',
#     'V37': 'Social Science: Labor Studies',
#     'V99': 'Social Science: Social Science N.E.C.'
#     },
#     'Public & Societal Benefit' : {
#     'W01': 'Public & Societal Benefit: Alliance/Advocacy',
#     'W02': 'Public & Societal Benefit: Management & Technical Assistance',
#     'W03': 'Public & Societal Benefit: Professional Societies, Associations',
#     'W05': 'Public & Societal Benefit: Research Institutes and Public Policy Analysis',
#     'W11': 'Public & Societal Benefit: Single Organization Support',
#     'W12': 'Public & Societal Benefit: Fund Raising and Fund Distribution',
#     'W19': 'Public & Societal Benefit: Support N.E.C.',
#     'W20': 'Public & Societal Benefit: Government and Public Administration',
#     'W22': 'Public & Societal Benefit: Public Finance, Taxation, Monetary Policy',
#     'W24': 'Public & Societal Benefit: Citizen Participation',
#     'W30': 'Public & Societal Benefit: Military, Veterans\' Organizations',
#     'W40': 'Public & Societal Benefit: Public Transportation Systems',
#     'W50': 'Public & Societal Benefit: Telecommunications',
#     'W60': 'Public & Societal Benefit: Financial Institutions',
#     'W61': 'Public & Societal Benefit: Credit Unions',
#     'W70': 'Public & Societal Benefit: Leadership Development',
#     'W80': 'Public & Societal Benefit: Public Utilities',
#     'W90': 'Public & Societal Benefit: Consumer Protection',
#     'W99': 'Public & Societal Benefit: Public, Society Benefit N.E.C.'
#     },
#     'Religion-Related': {
#     'X01': 'Religion-Related: Alliance/Advocacy',
#     'X02': 'Religion-Related: Management & Technical Assistance',
#     'X03': 'Religion-Related: Professional Societies & Associations',
#     'X05': 'Religion-Related: Research Institutes and Public Policy Analysis',
#     'X11': 'Religion-Related: Single Organization Support',
#     'X12': 'Religion-Related: Fund Raising and Fund Distribution',
#     'X19': 'Religion-Related: Support N.E.C.',
#     'X20': 'Religion-Related: Christianity',
#     'X21': 'Religion-Related: Protestant',
#     'X22': 'Religion-Related: Roman Catholic',
#     'X30': 'Religion-Related: Judaism',
#     'X40': 'Religion-Related: Islam',
#     'X50': 'Religion-Related: Buddhism',
#     'X70': 'Religion-Related: Hinduism',
#     'X80': 'Religion-Related: Religious Media, Communications',
#     'X81': 'Religion-Related: Religious Film, Video',
#     'X82': 'Religion-Related: Religious Television',
#     'X83': 'Religion-Related: Religious Printing, Publishing',
#     'X84': 'Religion-Related: Religious Radio',
#     'X90': 'Religion-Related: Interfaith Coalitions',
#     'X99': 'Religion-Related: Religion Related N.E.C.'
#     },
#     'Mutual/Membership Benefit': {
#     'Y01': 'Mutual/Membership Benefit: Alliance/Advocacy',
#     'Y02': 'Mutual/Membership Benefit: Management & Technical Assistance',
#     'Y03': 'Mutual/Membership Benefit: Professional Societies, Associations',
#     'Y05': 'Mutual/Membership Benefit: Research Institutes and Public Policy Analysis',
#     'Y11': 'Mutual/Membership Benefit: Single Organization Support',
#     'Y12': 'Mutual/Membership Benefit: Fund Raising and Fund Distribution',
#     'Y19': 'Mutual/Membership Benefit: Support N.E.C.',
#     'Y20': 'Mutual/Membership Benefit: Insurance Providers',
#     'Y22': 'Mutual/Membership Benefit: Local Benevolent Life Insurance Associations',
#     'Y23': 'Mutual/Membership Benefit: Mutual Insurance Company & Associations',
#     'Y24': 'Mutual/Membership Benefit: Supplemental Unemployment Compensation',
#     'Y25': 'Mutual/Membership Benefit: State-Sponsored Worker\'s Compensation Reinsurance Organizations',
#     'Y30': 'Mutual/Membership Benefit: Pension and Retirement Funds',
#     'Y33': 'Mutual/Membership Benefit: Teachers Retirement Fund Association',
#     'Y34': 'Mutual/Membership Benefit: Employee Funded Pension Trusts',
#     'Y35': 'Mutual/Membership Benefit: Multi-Employer Pension Plans',
#     'Y40': 'Mutual/Membership Benefit: Fraternal Societies',
#     'Y41': 'Mutual/Membership Benefit: Fraternal Beneficiary Societies',
#     'Y42': 'Mutual/Membership Benefit: Domestic Fraternal Societies',
#     'Y43': 'Mutual/Membership Benefit: Voluntary Employees Beneficiary Associations (Non-Government)',
#     'Y44': 'Mutual/Membership Benefit: Voluntary Employees Beneficiary Associations (Government)',
#     'Y50': 'Mutual/Membership Benefit: Cemeteries',
#     'Y99': 'Mutual/Membership Benefit: Mutual/Membership Benefit N.E.C.'
#     },
#     'Unknown' : {'Z99' : 'Unknown'}
# }
