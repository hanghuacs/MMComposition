var dom = document.getElementById("container");
var myChart = echarts.init(dom, null, {
  renderer: "canvas",
  useDirtyRect: false,
});
var app = {};

var option;

var data = [
  {
    name: "Perception",
    itemStyle: {
      color: "#84c3b7",
    },
    children: [
      {
        name: "Object\nPerception",
        value: 529,
        itemStyle: {
          color: "#e9eee0",
        },
      },
      {
        name: "Attribute\nPerception",
        value: 519,
        itemStyle: {
          color: "#ebf1b7",
        },
      },
      {
        name: "Relation\nPerception",
        value: 615,
        itemStyle: {
          color: "#e2ebcf",
        },
      },
      {
        name: "Text\nRendering",
        value: 56,
        itemStyle: {
          color: "#d5ead5",
        },
      },
      {
        name: "Counting\nPerception",
        value: 261,
        itemStyle: {
          color: "#d6e4bb",
        },
      },
      {
        name: "Visual\nSimilarity",
        value: 74,
        itemStyle: {
          color: "#c8d6cf",
        },
      },
      {
        name: "Difference\nSpotting",
        value: 241,
        itemStyle: {
          color: "#eef5d9",
        },
      },
    ],
  },
  {
    name: "Reasoning",
    itemStyle: {
      color: "#f2b56f",
    },
    children: [
      {
        name: "Object\nReasoning",
        value: 109,
        itemStyle: {
          color: "#ffcea2",
        },
      },
      {
        name: "Attribute\nReasoning",
        value: 222,
        itemStyle: {
          color: "#faeed3",
        },
      },
      {
        name: "Counting\nReasoning",
        value: 226,
        itemStyle: {
          color: "#fedec0",
        },
      },
      {
        name: "Object\nInteraction",
        value: 164,
        itemStyle: {
          color: "#fbe7d2",
        },
      },
      {
        name: "Relation\nReasoning",
        value: 867,
        itemStyle: {
          color: "#f8f3da",
        },
      },
    ],
  },
  {
    name: "Probing",
    itemStyle: {
      color: "#8fbcd4",
    },
    children: [
      {
        name: "VL\nComposition\nProbing",
        value: 459,
        itemStyle: {
          color: "#8fbcd4",
        },
      },
    ],
  },
];

var total = data.reduce((sum, current) => sum + current.children.reduce((cSum, cCurrent) => cSum + cCurrent.value, 0), 0);

option = {
  series: {
    type: "sunburst",
    data: data,
    radius: [0, "95%"],
    sort: null,
    emphasis: {
      focus: "ancestor",
    },
    startAngle: 180, // Adjust start angle to split horizontally
    label: {
      textStyle: {
        fontSize: 15,
        fontFamily: "Times New Roman",
      },
      formatter: function (params) {
        var percentage = params.value //((params.value / total) * 100).toFixed(2) + "%";
        return `${params.name}\n{small|${percentage}}`;
      },
      rich: {
        small: {
          fontSize: 15,
          fontFamily: "Blod",
          lineHeight: 15,
        },
      },
    },
    levels: [
      {},
      {
        r0: "16%",
        r: "40%",
        itemStyle: {
          borderRadius: 6,
          borderWidth: 4,
        },
        label: {
          rotate: "tangential",
        },
      },
      {
        r0: "40%",
        r: "75%",
        itemStyle: {
          borderRadius: 6,
          borderWidth: 4,
        },
        label: {
          align: "right",
        },
      },
      {
        r0: "75%",
        r: "77%",
        label: {
          position: "outside",
          padding: 3,
          silent: false,
        },
        itemStyle: {
          borderColor: "transparent",
          borderWidth: 0,
        },
      },
    ],
  },
};

if (option && typeof option === "object") {
  myChart.setOption(option);
}

window.addEventListener("resize", myChart.resize);
