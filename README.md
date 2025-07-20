# Money Laundering Detection using Graph Neural Network (GNN)

Proyek ini dikembangkan untuk mengikuti lomba **Datathon 2025**, dengan tujuan utama mendeteksi transaksi pencucian uang (money laundering) dari transaksi finansial menggunakan pendekatan **Graph Neural Network (GNN)**. Model yang digunakan adalah **Graph Convolutional Network (GCN)** yang diimplementasikan dengan PyTorch Geometric.

## Tujuan
Mendeteksi pola mencurigakan pada transaksi berdasarkan hubungan antara akun pengirim dan penerima dalam bentuk graf. GNN dipilih karena kekuatannya dalam memodelkan hubungan antar entitas secara struktural.

## Struktur Dataset
Dataset terdiri dari:
- Fitur numerik pada node (akun)
- Informasi edge (hubungan pengirim–penerima transaksi)
- Label target: `0` = Legitimate, `1` = Money Laundering

## 🔧 Teknologi & Tools
- Python 3.x
- PyTorch & PyTorch Geometric
- Scikit-learn
- Matplotlib (visualisasi confusion matrix)

## ⚙️ Arsitektur Model
```python
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
