import styled from 'styled-components';

const InputContainer = styled.div`
  position: absolute;
  left: ${({ style }) => style.x}px;
  top: ${({ style }) => style.y}px;
`;

const CheckboxContainer = styled.div`
  position: absolute;
  left: ${({ style }) => style.x}px;
  top: ${({ style }) => style.y}px;
`;

const TextContainer = styled.div`
  position: absolute;
  left: ${({ style }) => style.x}px;
  top: ${({ style }) => style.y}px;
  font-family: ${({ style }) => style.font};
  font-size: ${({ style }) => style.fontSize}px;
  color: ${({ style }) => style.color};
`;