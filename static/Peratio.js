
function myFunction1(){
    var num1 = document.querySelector(".net-income").value;
    var num2 = document.querySelector(".Total-shares").value;
    var num3 = document.querySelector(".current-share").value;
    var earnpershare=parseFloat(num1)/parseFloat(num2);
    var ans=parseFloat(num3)/parseFloat(earnpershare);
    ans=ans.toFixed(2);
    document.querySelector(".output1").innerHTML = "Your Price-to-Earnings Ratio is "+ans;
}

function myFunction(){
    var num1 = document.querySelector(".vfinal").value;
    var num2 = document.querySelector(".vinitial").value;
    var num3 = document.querySelector(".time").value;
    var earnpershare=parseFloat(num1)/parseFloat(num2);
    var temp=parseFloat(1)/parseFloat(num3);
    var ans=Math.pow(earnpershare,temp);
    ans--;
    ans=(ans*100);
    ans=ans.toFixed(2);
    document.querySelector(".output").innerHTML = "Your Compound Annual Growth Rate is "+ans+"%";
}

function myFunction2(){
    var num1 = document.querySelector(".currprice").value;
    var num2 = document.querySelector(".iniprice").value;
    // var num3 = document.querySelector(".time").value;
    var ans=((parseFloat(num1)-parseFloat(num2))/parseFloat(num2))*100;
    // num3--;
    ans=ans.toFixed(2);
    document.querySelector(".output2").innerHTML = "Your return for investment is "+ans;
}

function myFunction4(){
    var num1 = document.querySelector(".entry").value;
    var num2 = document.querySelector(".stop").value;
    var num3 = document.querySelector(".profit").value;
    var ans=parseFloat(num1)-parseFloat(num2);
    var ans1=parseFloat(num3)-parseFloat(num1);
    var ans3=ans/ans1;
    var ans4=ans3.toFixed(2);
    document.querySelector(".output4").innerHTML = "Your risk reward ratio is "+ans4;
}