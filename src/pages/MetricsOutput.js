import * as React from "react";
import '../CSS/MetricsOutput.css';
import { useNavigate } from "react-router-dom";
import { getResults, getConfigs, setRequirements } from "../setConfigs";
import { Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Dialog, DialogActions, DialogTitle, Button } from "@mui/material";
import { CircularProgress, Box } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import axios from "axios";

const MetricsOutput = () => {

    var results = getResults()

    const navigate = useNavigate();

    let configs = getConfigs();

    const [rows, setRows] = React.useState([])
    const [open, setOpen] = React.useState(false)

    const [isLoading, setIsLoading] = React.useState(false);
    const [done, setDone] = React.useState(false);

    const [loading, setLoading] = React.useState(false);
    const [isDone, setIsDone] = React.useState(false);

    React.useEffect(() => {
        const assignValues = async () => {

            if (results.length === 1) {
                setRows([{ metric: "Accuracy", valueR1: results[0][0], valueR2: 1 },
                { metric: "MSLE", valueR1: results[0][1], valueR2: 1 },
                { metric: "Precision", valueR1: results[0][2], valueR2: 1 },
                { metric: "Recall", valueR1: results[0][3], valueR2: 1 },
                { metric: "F1-Measure", valueR1: results[0][4], valueR2: 1 },
                { metric: "Kappa", valueR1: results[0][5], valueR2: 1 }])

            } else {
                setRows([{ metric: "Accuracy", valueR1: results[0][0], valueR2: results[1][0] },
                { metric: "MSLE", valueR1: results[0][1], valueR2: results[1][1] },
                { metric: "Precision", valueR1: results[0][2], valueR2: results[1][2] },
                { metric: "Recall", valueR1: results[0][3], valueR2: results[1][3] },
                { metric: "F1-Measure", valueR1: results[0][4], valueR2: results[1][4] },
                { metric: "Kappa", valueR1: results[0][5], valueR2: results[1][5] }])
            }
        }
        assignValues()
    })


    const continueflg = () => {

        if (results[0][6] === "100.0%") {
            setOpen(true)
        } else {
            let nextBatch = [configs[0], configs[1], configs[2]]

            setIsLoading(true)

            axios.post('http://localhost:8000/runNextTrainBatchScript', { content: nextBatch })
                .then((response) => {
                    setRequirements(response.data.text)
                    setDone(true)
                    setTimeout(() => {
                        handleCloseLoading()
                    }, 2000)
                    setTimeout(() => {
                        navigate("/configs/req-input/test-set-labeling/train-batch-labeling");
                    }, 1000)
                })
                .catch((e) => {
                    console.log('Upload Error')
                })
        }
    }

    const stopflg = () => {

        let nextBatch = [configs[0], configs[2]]

        setLoading(true)

        axios.post('http://localhost:8000/runFinalClassification', { content: nextBatch })
            .then((response) => {
                setIsDone(true)
                setTimeout(() => {handleCloseEnd()}, 2000)
                setTimeout(() => {
                    navigate("/configs/req-input/test-set-labeling/train-batch-labeling/stopFlg/downloadFile");}, 1000)
            })
            .catch((e) => {
                console.log('Upload Error')
            })
    }

    const returnHome = () => {
        navigate("/configs");
    }

    const handleClose = () => {
        setOpen(false)
    }

    const handleCloseLoading = () => {
        setIsLoading(false)
    }

    const handleCloseEnd = () => {
        setLoading(false)
    }


    return (
        <div className="UI">
            <div className='Header'>
                <div className="Logo">
                    <img src={require('../imgs/logotipo-removebg-preview.png')} className="logo" onClick={returnHome} />
                </div>
            </div>,

            <div className='ResultsOut'>

                <h1 className='Title' >Training Results</h1>

                <Typography variant="h6" gutterBottom paddingTop="25px" paddingBottom="15px">The model was able to reach these results, using <span style={{ fontWeight: "bold" }}>{results[0][6]}</span> of the dataset.</Typography>

                <div className='ResultsTable'>
                    <TableContainer component={Paper} sx={{ width: "55%" }} >
                        <Table aria-label="simple table">
                            <TableHead>
                                <TableRow style={{ backgroundColor: "#93c47d", color: "#FFFFFF" }}>
                                    <TableCell align="left" sx={{ fontWeight: "bold" }}>Metric</TableCell>
                                    {results.length === 2 ? <TableCell align="left" sx={{ fontWeight: "bold" }}>Previous Run</TableCell> : null}
                                    <TableCell align="left" sx={{ fontWeight: "bold" }}>Current Run</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {rows.map((item) => (

                                    <TableRow
                                        key={item.metric}
                                        sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
                                    >
                                        <TableCell component="th" scope="row">
                                            {item.metric}
                                        </TableCell>
                                        {results.length === 2 ? <TableCell align="left">{item.valueR2}</TableCell> : null}
                                        <TableCell align="left">{item.valueR1}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>

                </div>

                {results[0][6] !== "100,0%" ? <Typography variant="h6" paddingTop="20px">Do you wish to continue with the training or stop and classify the remaining requirements?</Typography> : <Typography variant="h6" align="center" paddingTop="20px" sx={{ fontWeight: "bold" }}>The Classification has ended!</Typography>}

                <br />
                <br />

                <button onClick={continueflg} className='ContinueButtom'>
                    Continue
                </button>

                <button onClick={stopflg} className='StopButtom'>
                    Stop
                </button>

                <Dialog open={open} onClose={handleClose} maxWidth="lg">
                    <DialogTitle>
                        {"100% of the dataset has already been used, please press \"Stop\""}
                    </DialogTitle>
                    <DialogActions>
                        <Button
                            variant="contained"
                            style={{ backgroundColor: "#0e3c45", textTransform: "none" }}
                            onClick={handleClose}
                            autoFocus
                        >
                            OK
                        </Button>
                    </DialogActions>
                </Dialog>

                <Dialog open={isLoading} onClose={handleCloseLoading} maxWidth="lg" >
                    <DialogTitle alignItems="center">
                        {done ? <CheckCircleIcon sx={{ fontSize: 200, paddingLeft: "50px" }} color="success" /> : (<Box sx={{ paddingLeft: "50px" }}>
                            <CircularProgress size={200} />
                        </Box>)}
                    </DialogTitle>
                    <DialogTitle>
                    {"Training Machine Learning Model..."}
                    </DialogTitle>
                </Dialog>

                <Dialog open={loading} onClose={handleCloseEnd} maxWidth="lg" >
                    <DialogTitle alignItems="center">
                        {isDone ? <CheckCircleIcon sx={{ fontSize: 200, paddingLeft: "50px" }} color="success" /> : (<Box sx={{ paddingLeft: "50px" }}>
                            <CircularProgress size={200} />
                        </Box>)}
                    </DialogTitle>
                    <DialogTitle>
                    {"Preparing the Final Classified File..."}
                    </DialogTitle>
                </Dialog>
            </div>
        </div>
    )
}

export default MetricsOutput;