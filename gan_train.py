import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import glob
from music21 import converter, instrument, note, chord
from typing import Tuple

class LSTMGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0.squeeze(0), c0.squeeze(0)))
        out = self.fc(out[:, -1, :])
        return out

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0.squeeze(0), c0.squeeze(0)))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

class MusicLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        notes = []
        durations = []
        offsets = []

        for file in glob.glob(self.folder_path + "/*.mid"):
            try:
                midi = converter.parse(file)
                print("Parsing %s" % file)

                notes_to_parse = None

                try: # file has instrument parts
                    s2 = instrument.partitionByInstrument(midi)
                    notes_to_parse = s2.parts[0].recurse()
                except: # file has notes in a flat structure
                    notes_to_parse = midi.flat.notes

                offset_base = 0
                for element in notes_to_parse:
                    is_note_or_chord = False

                    if isinstance(element, note.Note):
                        pitch_number = element.pitch.midi
                        notes.append(pitch_number)
                        is_note_or_chord = True
                    elif isinstance(element, chord.Chord):
                        pitches = [n.pitch.midi for n in element.normalOrder]
                        notes.append(pitches)
                        is_note_or_chord = True

                    if is_note_or_chord:
                        offsets.append(element.offset - offset_base)
                        durations.append(element.quarterLength)
                        is_note_or_chord = False
                        offset_base = element.offset
            except Exception as e:
                print(f"Error parsing file {file}: {e}")
                continue

        # Convert lists to PyTorch tensors
        notes_tensor = torch.tensor(notes, dtype=torch.float)
        durations_tensor = torch.tensor(durations, dtype=torch.float)
        offsets_tensor = torch.tensor(offsets, dtype=torch.float)

        return notes_tensor, durations_tensor, offsets_tensor


class MIDIDataset(Dataset):
    def __init__(self, notes, durations, offsets):
        self.notes = torch.tensor(notes, dtype=torch.float)
        self.durations = torch.tensor(durations, dtype=torch.float)
        self.offsets = torch.tensor(offsets, dtype=torch.float)

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        return self.notes[idx], self.durations[idx], self.offsets[idx]


def train_network(generator, discriminator, train_loader, num_epochs, lr_gen, lr_disc, device):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam(generator.parameters(), lr=lr_gen)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)

    best_gen_loss = float('inf')
    best_disc_loss = float('inf')
    best_gen_state = None
    best_disc_state = None

    for epoch in range(num_epochs):
        for i, (notes, durations, offsets) in enumerate(train_loader):
            notes, durations, offsets = notes.to(device), durations.to(device), offsets.to(device)

            # Train Discriminator
            optimizer_disc.zero_grad()
            
            # Concatenate notes, durations, and offsets along the second dimension
            real_samples = torch.cat((notes.unsqueeze(1),
                                      durations.unsqueeze(1),
                                      offsets.unsqueeze(1)), dim=1)

            batch_size = real_samples.size(0)
            h0 = torch.zeros(2, batch_size, discriminator.hidden_size).to(device)
            c0 = torch.zeros(2, batch_size, discriminator.hidden_size).to(device)
            real_outputs = discriminator(real_samples)
            
            # Generate fake samples
            fake_samples = generator(torch.randn(batch_size, 100).to(device))
            fake_outputs = discriminator(fake_samples)

            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            disc_loss_real = criterion(real_outputs, real_labels)
            disc_loss_fake = criterion(fake_outputs, fake_labels)
            disc_loss = disc_loss_real + disc_loss_fake
            disc_loss.backward()
            optimizer_disc.step()

            # Train Generator
            optimizer_gen.zero_grad()
            fake_samples = generator(torch.randn(batch_size, 100).to(device))
            fake_outputs = discriminator(fake_samples)
            gen_loss = criterion(fake_outputs, real_labels)
            gen_loss.backward()
            optimizer_gen.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}')
        
        # Update best model states and losses
        if gen_loss < best_gen_loss:
            best_gen_loss = gen_loss
            best_gen_state = generator.state_dict()
        if disc_loss < best_disc_loss:
            best_disc_loss = disc_loss
            best_disc_state = discriminator.state_dict()
    
    return best_gen_state, best_disc_state, best_gen_loss.item(), best_disc_loss.item()


folder_path = "Music"
notes, durations, offsets = MusicLoader(folder_path).load_data()
dataset = MIDIDataset(notes, durations, offsets)

# Create data loader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Initialize Generator and Discriminator
generator = LSTMGenerator(input_size=100, hidden_size=256, output_size=3, num_layers=2)  # Update output_size to 3
discriminator = LSTMDiscriminator(input_size=3, hidden_size=256, num_layers=2)

# Training parameters
num_epochs = 100
lr_gen = 0.001
lr_disc = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the network
best_gen_state, best_disc_state, gen_loss, disc_loss = train_network(generator, discriminator, train_loader, num_epochs, lr_gen, lr_disc, device)

# Save the best performing model
if not os.path.exists('models'):
    os.makedirs('models')

# Save generator and discriminator with their loss values in the file names
torch.save(best_gen_state, f'models/generator_loss_{gen_loss:.4f}.pth')
torch.save(best_disc_state, f'models/discriminator_loss_{disc_loss:.4f}.pth')