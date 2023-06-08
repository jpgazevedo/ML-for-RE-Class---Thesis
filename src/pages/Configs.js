import * as React from "react";
import '../CSS/Configs.css';
import { setConfigs, clearResults } from "../setConfigs";
import { useNavigate } from "react-router-dom";

const Configs = () => {

    const navigate = useNavigate();

    const optionsRE = [
        { label: 'Natural Text Requirements', value: 'RE' },
        { label: 'User Stories', value: 'US' },
    ]

    const optionsML = [
        { label: 'Multinomial Naïve Bayes', value: 'MNB' },
        { label: 'Support Vector Machine', value: 'SVC' },
        { label: 'Logistic Regression', value: 'LR' },
        { label: 'Neural Network', value: 'NN' },
    ]

    const optionsAL = [
        { label: 'Least Confident', value: 'LC' },
        { label: 'Entropy Measure', value: 'EM' },
        { label: 'Maring Sampling', value: 'MS' },
    ]

    const [valueRE, setREValue] = React.useState('RE');

    const [valueML, setMLValue] = React.useState('MNB');

    const [valueAL, setALValue] = React.useState('LC');

    const handleMLChange = (event) => {
        setMLValue(event.target.value);
    };

    const handleALChange = (event) => {
        setALValue(event.target.value);
    };

    const handleREChange = (event) => {
        setREValue(event.target.value);
    };

    const next = () => {
        setConfigs(valueML, valueAL, valueRE, 'Y');
        clearResults();
        navigate("/configs/req-inputConfigs");
    }

    const returnHome = (event) => {
        navigate("/configs");
    }

    return (
        <div className="UI">
            <div className='Header'>
                <div className="Logo">
                    <img src={require('../imgs/logotipo-removebg-preview.png')} className="logo" onClick={returnHome} />
                </div>
            </div>,

            <div className="Confs">
                <h1 className='TitleConfs'>Classification Configuration</h1>
                <p className="DoptTitle">Select the Machine Learning Classifier</p>

                <select value={valueML} onChange={handleMLChange} className='ALDROP'>
                    {optionsML.map((option) => (
                        <option value={option.value}>{option.label}</option>
                    ))}
                </select>

                <p className="DoptTitle">Select the Active Learning Strategy</p>
                <select value={valueAL} onChange={handleALChange} className='ALDROP'>
                    {optionsAL.map((option) => (
                        <option value={option.value}>{option.label}</option>
                    ))}
                </select>

                <p className="DoptTitle">Select the Type of Requirements</p>
                <select value={valueRE} onChange={handleREChange} className='ALDROP'>
                    {optionsRE.map((option) => (

                        <option value={option.value}>{option.label}</option>

                    ))}
                </select>

                <p />
                <h1 />
                <p />
                <h1 />

                <button onClick={next} className='ConfirmConfs'>
                    Confirm
                </button>
            </div>
        </div>
    )
}

export default Configs;