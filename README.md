# GraphDTA: Dự đoán Ái lực Liên kết Thuốc – Đích  
*(Drug–Target Affinity Prediction)*

Dự án này triển khai mô hình **GraphDTA** (Graph Neural Networks for Drug–Target Affinity), sử dụng **mạng nơ-ron đồ thị (GNN)** để dự đoán ái lực liên kết giữa:
- **Thuốc**: biểu diễn dưới dạng **đồ thị phân tử**
- **Protein đích**: biểu diễn dưới dạng **chuỗi amino acid**

---

## 📋 Tài nguyên & Cấu trúc dự án

Repository bao gồm mã nguồn và dữ liệu cần thiết để:
- tái lập kết quả huấn luyện
- chạy ứng dụng demo dự đoán

```
.
├── models/
│   ├── ginconv.py
│   ├── gat.py
│   ├── gcn.py
│   └── ...
├── data/
│   ├── raw/
│   └── processed/
├── create_data.py
├── training.py
├── frontend.py
├── utils.py
└── README.md
```

### 📁 Mô tả các thành phần chính

- **`models/`**  
  Chứa định nghĩa các mô hình GNN:
  - `GINConvNet`
  - `GATNet`
  - `GAT_GCN`
  - `GCNNet`

- **`data/`**  
  Chứa hai bộ dữ liệu benchmark chuẩn:
  - **Davis**
  - **Kiba**

- **`create_data.py`**  
  Script chuyển dữ liệu thô (SMILES, protein sequence) sang định dạng **đồ thị PyTorch Geometric (.pt)**

- **`training.py`**  
  Script chính để huấn luyện mô hình

- **`frontend.py`**  
  Ứng dụng web (Streamlit) để test và trực quan hóa mô hình

- **`utils.py`**  
  Các hàm hỗ trợ:
  - Tính toán metrics (MSE, CI)
  - Xử lý và chuẩn hóa dữ liệu

---

## 🛠️ Cài đặt môi trường

Khuyến nghị sử dụng **Conda** để quản lý môi trường.

### 1 Tạo môi trường Conda

```bash
conda create -n graphdta python=3.8
conda activate graphdta
```

### 2 Cài đặt các thư viện cần thiết

Dự án yêu cầu:
- PyTorch
- PyTorch Geometric
- RDKit
- Streamlit

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install -c conda-forge rdkit

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

pip install streamlit pandas numpy networkx Pillow
```

---

## 🚀 Hướng dẫn chạy

### 🔹 Bước 1: Chuẩn bị dữ liệu

Chuyển đổi dữ liệu gốc (SMILES & Protein Sequence) sang định dạng đồ thị `.pt`:

```bash
python create_data.py
```

Sau khi chạy xong, các file sau sẽ được tạo trong `data/processed/`:

```
davis_train.pt
davis_test.pt
kiba_train.pt
kiba_test.pt
```

---

### 🔹 Bước 2: Huấn luyện mô hình (Training)

Sử dụng script `training.py` với cú pháp:

```bash
python training.py
```

#### Tham số

- `dataset_index`  
  - `0`: Davis  
  - `1`: Kiba  

- `cuda_index`  
  - `0`, `1`: chọn GPU  
  - Nếu không có GPU → tự động chạy CPU

#### Ví dụ

Huấn luyện trên tập **Davis**:

```bash
python training.py 0 0 
```

Sau khi huấn luyện xong, mô hình tốt nhất sẽ được lưu dưới dạng:

```
model_GINConvNet_davis.model
```

---

### 🔹 Bước 3: Chạy Demo Dự đoán (Inference App)

Chạy ứng dụng web để dự đoán ái lực từ dữ liệu đầu vào:

```bash
streamlit run frontend.py
```

Trình duyệt sẽ tự động mở giao diện.  
Bạn có thể:
- Chọn dataset (Davis / Kiba)
- Nhập **SMILES**
- Nhập **Protein Sequence**
- Xem kết quả dự đoán **pKd / KIBA score**

---

## 📊 Kết quả thực nghiệm

Hiệu năng của mô hình **GINConvNet** trên hai bộ dữ liệu benchmark:

| Dataset | Model       | MSE ↓ | CI ↑ |
|--------|------------|-------|------|
| Davis  | GINConvNet | 0.228 | 0.893 |
| Kiba   | GINConvNet | 0.164 | 0.874 |

**Chỉ số đánh giá:**
- **MSE (Mean Squared Error)**: càng thấp càng tốt
- **CI (Concordance Index)**: càng cao càng tốt

---
