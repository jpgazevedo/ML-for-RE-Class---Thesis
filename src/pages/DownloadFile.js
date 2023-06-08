import * as React from "react";
import '../CSS/DownloadFile.css';
import { useNavigate } from "react-router-dom";

const DownloadFile = () => {

    const navigate = useNavigate();


    const returnHome = () => {
        navigate("/configs");
    }

    const handleDownload = () => {
        const link = document.createElement('a');

        link.download = 'Classified_Dataset.txt';

        link.href = require('../files/Classified_Dataset.txt')

        link.click();
    }

    return (
        <div className="UI">
            <div className='Header'>
                <div className="Logo">
                    <img src={require('../imgs/logotipo-removebg-preview.png')} className="logo" onClick={returnHome} />
                </div>
            </div>,
            <div className="Download">
                <h1>Download File</h1>
                <p>Click in the following button to download the final file:</p>
                <button href="./760.js" download="file" className='ConfirmConfs' onClick={handleDownload}>
                    Download
                </button>
            </div>
        </div>
    )
}

export default DownloadFile;