from pickle import load

import gradio as gr

from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import MolFromSmiles, Mol

from utils import calc_morgan

with open('./src/models/models_engine/model_engine.pkl', 'rb') as file:
    model = load(file)


def predict(smiles: str) -> str:
    mol = MolFromSmiles(smiles)
    if not mol:
        return ['Incorrect SMILES', MolToImage(Mol())]
    mol_img = MolToImage(mol)

    mol = calc_morgan(mol)
    prediction = round(model.predict([mol])[0], 3)

    return [prediction, mol_img]


ex_mol = MolFromSmiles('CCO')
ex_img = MolToImage(ex_mol)

clean_css = """
footer { display: none !important; }
footer:has(.built-with) { display: none !important; }
"""

with gr.Blocks(
    title="Citotox"
) as app:
    gr.Markdown('# Simple program for cytotoxicity prediction')

    with gr.Row():
        with gr.Column():
            smiles_input = gr.Textbox(
                label='Smiles',
                value='',
                placeholder='Example: CCO',
                submit_btn='prediction'
            )
            predict_output = gr.Textbox(
                label='Cytotoxicity prediction',
                placeholder=predict('CCO')[0]
            )

        molecule_image = gr.Image(
            value=ex_img,
            width=200,
        )

        smiles_input.submit(
            fn=predict,
            inputs=[smiles_input],
            outputs=[predict_output, molecule_image]   
        )

if __name__ == '__main__':
    app.launch(
        server_name="0.0.0.0",
        server_port=5000,
        css=clean_css
    )