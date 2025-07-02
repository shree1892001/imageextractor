interface TextElementProps {
  id: string;
  type: string;
  content: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  properties: {
    font: string;
    fontSize: number;
    color: number;
    flags: number;
  };
}