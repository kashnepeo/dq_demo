Common = {};
Common.Date = {}
Common.Date.getReq_time = function(req_time, dateSep) {
    var temp = new Date(req_time * 1000);
    var year = temp.getFullYear();
    var month = temp.getMonth() < 10 ? "0" + (temp.getMonth() + 1) : (temp.getMonth() + 1);
    var day = temp.getDate();

    var time = temp.toTimeString().substring(0,8);
    // var hour = temp.getHours();
    // var minutes = temp.getMinutes();
    // var second = temp.getSeconds();

    return year + dateSep + month + dateSep + day + " " + time;
}