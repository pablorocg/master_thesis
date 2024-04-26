# Description: Script para descargar los datos de HCP_105 desde Zenodo y descomprimirlos en la carpeta HCP_105
# Usage: bash download_HCP_105_data.sh
# --------------------------
echo "Downloading HCP_105 data from Zenodo..."

# Asegurarse de que la carpeta HCP_105 existe
mkdir -p HCP_105

# Reemplazar 'your_file.zip' con el nombre real del archivo que deseas descargar
wget -O HCP_105/HCP105_Zenodo_NewTrkFormat.zip https://zenodo.org/records/1477956/files/HCP105_Zenodo_NewTrkFormat.zip

echo "Extracting files..."
unzip HCP_105/HCP105_Zenodo_NewTrkFormat.zip -d HCP_105

# Limpiar el archivo zip descargado
rm HCP_105/HCP105_Zenodo_NewTrkFormat.zip

# Mostrar un mensaje
echo "HCP_105 data downloaded and extracted in HCP_105 folder"

