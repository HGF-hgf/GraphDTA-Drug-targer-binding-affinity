import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
import os

st.set_page_config(page_title="GraphDTA Predictor", layout="wide")

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
max_seq_len = 1000

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                           'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                                           'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict.get(ch, 0)
    return x

class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        nn1 = nn.Sequential(nn.Linear(num_features_xd, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = nn.Linear(dim, output_dim)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINConvNet().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, device, None
        except Exception as e:
            return None, device, str(e)
    return None, device, "File not found"


st.title("💊 GraphDTA: Drug-Target Affinity Prediction")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    dataset_choice = st.selectbox(
        "Select Model Type (Dataset):",
        ("Davis", "Kiba")
    )
    
    default_filename = f"model_GINConvNet_{dataset_choice.lower()}.model"
    
    model_file = st.text_input("Model File Path:", value=default_filename)
    
    # Load model
    model, device, error_msg = load_model(model_file)
    
    st.divider()
    if model:
        st.success(f"✅ Model **{dataset_choice}** Loaded!")
        st.info(f"Device: `{device}`")
    else:
        st.error(f"❌ Error loading model: {model_file}")
        if error_msg:
            st.caption(f"Details: {error_msg}")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.subheader("1. Drug Input (SMILES)")
    default_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    drug_smiles = st.text_area("Enter SMILES string:", value=default_smiles, height=150)

with input_col2:
    st.subheader("2. Target Input (Sequence)")
    default_seq = "PFWKILNPLLERGTYYYFMGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVEHEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
    target_seq = st.text_area("Enter Protein Sequence:", value=default_seq, height=150)
    st.caption(f"Sequence Length: {len(target_seq)}")

st.markdown("---")

vis_col, action_col = st.columns([1, 1])

with vis_col:
    st.subheader("3. Structure Visualization")
    if drug_smiles:
        try:
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol:
                st.image(Draw.MolToImage(mol, size=(400, 300)), caption="Chemical Structure of Drug", width=400)
            else:
                st.warning("⚠️ Invalid SMILES string. Cannot visualize.")
        except:
            st.error("Error generating structure image.")
    else:
        st.info("Enter SMILES to see structure.")
        
with action_col:
    st.subheader("4. Prediction Action")
    
    predict_btn = st.button("🚀 Predict Affinity", type="primary", use_container_width=True)
    
    st.markdown("### Results")
    
    if predict_btn:
        if not model:
            st.error("Please load a valid model first!")
        elif not drug_smiles or not target_seq:
            st.warning("Please input both SMILES and Sequence.")
        else:
            with st.spinner('Analyzing interaction...'):
                try:
                    # 1. Preprocess
                    graph_data = smile_to_graph(drug_smiles)
                    if graph_data is None:
                        st.error("Cannot convert SMILES to Graph.")
                        st.stop()
                    
                    c_size, features, edge_index = graph_data
                    xt = seq_cat(target_seq)
                    
                    # 2. Tensorize
                    data = Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([0]))
                    data.target = torch.LongTensor([xt])
                    data.__setitem__('c_size', torch.LongTensor([c_size]))
                    batch_data = Batch.from_data_list([data]).to(device)
                    
                    # 3. Inference
                    with torch.no_grad():
                        prediction = model(batch_data).item()
                    
                    st.success("Prediction Complete!")
                    
                    # 4. Display Logic (Davis vs Kiba)
                    with st.container(border=True):
                        col_res_1, col_res_2 = st.columns(2)
                        
                        if dataset_choice == "Davis":
                            # DAVIS LOGIC: Higher pKd = Stronger
                            with col_res_1:
                                st.metric(label="Predicted pKd", value=f"{prediction:.4f}")
                            # with col_res_2:
                            #     st.markdown("**Davis Interpretation:**")
                            #     st.caption("Unit: pKd (-logM). Higher is Better.")
                            #     if prediction < 5:
                            #         st.markdown("🔴 **Low** Affinity")
                            #     elif prediction < 7:
                            #         st.markdown("🟡 **Moderate** Affinity")
                            #     else:
                            #         st.markdown("🟢 **High** Affinity")
                                st.latex(r"K_d \approx 10^{-" + f"{prediction:.2f}" + r"} \text{ M}")

                        elif dataset_choice == "Kiba":
                            # KIBA LOGIC: Lower Score = Stronger
                            with col_res_1:
                                st.metric(label="Predicted KIBA Score", value=f"{prediction:.4f}")
                            # with col_res_2:
                            #     st.markdown("**Kiba Interpretation:**")
                            #     st.caption("Unit: KIBA Score. Lower is Better.")
                            #     if prediction > 3.0:
                            #         st.markdown("🔴 **Low** Affinity")
                            #     else:
                            #         st.markdown("🟢 **High** Affinity")
                            #     st.write("*(Note: KIBA score < 3.0 often indicates effective binding)*")

                except Exception as e:
                    st.error(f"Error: {e}")