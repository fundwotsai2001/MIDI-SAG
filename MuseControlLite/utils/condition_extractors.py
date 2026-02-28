import torch.nn.functional as F
import torch.nn as nn

class MelodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=129, embedding_dim=48)

        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(384, 384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(384, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

    def forward(self, melody_idxs):
        # melody_idxs: LongTensor of shape (B, 8, 4096)
        B, eight, L = melody_idxs.shape  # L == 4096

        # 1) Embed:
        #    (B, 8, 4096) → (B, 8, 4096, 48)
        embedded = self.embed(melody_idxs)

        # 2) Permute & reshape → (B, 8*48, 4096) = (B, 384, 4096)
        x = embedded.permute(0, 1, 3, 2)      # (B, 8, 48, 4096)
        x = x.reshape(B, eight * 48, L)       # (B, 384, 4096)

        # 3) Conv1 → (B, 384, 4096)
        x = F.silu(self.conv1(x))

        # 4) Conv2 → (B, 768, 4096)
        x = F.silu(self.conv2(x))

        # 5) Conv3 → (B, 768, 4096)
        x = F.silu(self.conv3(x))
        return x
class Chord_extractor(nn.Module):
    def __init__(self):
        super(Chord_extractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(12, 64, kernel_size=3, padding=1)  
        self.conv1d_2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(64, 256, kernel_size=3, padding=1)  
    def forward(self, x):
        x = self.conv1d_1(x)# shape: (batchsize, 128, 4756)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 256, 2378)
        x = F.silu(x)
        return x