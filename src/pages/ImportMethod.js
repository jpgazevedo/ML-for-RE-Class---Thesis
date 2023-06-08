import * as React from "react";
import '../CSS/ImportMethod.css';
import { useNavigate } from "react-router-dom";


const ImportMethod = () => {

    const navigate = useNavigate();

    const next = () => {
        if (importMethod.toString() === "MAN") {
            navigate("/configs/req-inputConfigs/manualInput");
        }
        else {
            navigate("/configs/req-inputConfigs/fileInput");
        }
    }

    const returnHome = () => {
        navigate("/configs");
    }

    const [importMethod, setUpdateMethod] = React.useState('MAN');

    const updateImportMethod = (event) => {
        setUpdateMethod(event.target.value);
    };

    return (
        <div className="UI">
            <div className='Header'>
                <div className="Logo">
                    <img src={require('../imgs/logotipo-removebg-preview.png')} className="logo" onClick={returnHome} />
                </div>
            </div>

            <div className="InputConfigs">
                <h1>Input Method Configuration</h1>
                <p>Please select if you pretend to manual insert your dataset, or through text file import.<br /> Note that the text file needs to be in proper format.</p>

                <div className="InputMethod" onChange={updateImportMethod}>
                    <input type="radio" value="MAN" name="gender" /> Manual Inserting
                    <br />
                    <input type="radio" value="IMP" name="gender" /> File Upload
                </div>
                <button onClick={next} className='ConfirmConfs'>
                    Confirm
                </button>
            </div>
        </div>
    )
}

export default ImportMethod;