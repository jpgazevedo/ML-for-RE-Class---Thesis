export const setConfigs = (mlc, als, tr, flg) => {
    localStorage.setItem("ML", mlc);
    localStorage.setItem("ALS", als);
    localStorage.setItem("TR", tr);
    localStorage.setItem("Flg", flg)
}

export const setRequirements = (text) => {

    var arr = text

    localStorage.setItem("requirements", arr);
}

export const setResults = (text) => {
    var arr = []
    arr.push(text.split(';'))
    var results = getResults()

    var auxArr = []
    var finalArr = []

    if (results !== null) {
        finalArr.push(results[0])
    }

    arr[0].map((result, index) => {
        if (index !== 1 && index !== 7) {
            auxArr.push(result + '%');
        } else {
            auxArr.push(result)
        }
    })

    if (finalArr.length === 0) {
        finalArr.push(auxArr)
    } else {
        finalArr.unshift(auxArr)
    }

    if (finalArr.length === 3) {
        finalArr.pop()
    }

    localStorage.setItem("Results", JSON.stringify(finalArr))
}

export const getResults = () => {
    var arr = JSON.parse(localStorage.getItem("Results"));

    return arr;
}

export const getConfigs = () => {
    return [localStorage.getItem("ML"), localStorage.getItem("ALS"), localStorage.getItem("TR"), localStorage.getItem("Flg")] || null;
}

export const getRequirements = () => {

    var arr = localStorage.getItem("requirements").split(';');

    arr.pop()

    return arr;
}

export const clearResults = () => {
    localStorage.removeItem("Results");
}



